import torch
import random
import numpy as np
from einops import rearrange

class RetrievalContextManager:
    def __init__(self, num_envs, config, latent_dim, device="cuda"):
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.enabled = bool(config.get("enable", True))
        self.threshold = float(config.get("threshold", 1.0))
        self.context_length = int(config.get("context_length", 8))
        self.max_bucket_size = int(config.get("max_bucket_size", 512))
        
        self.trigger_mode = config.get("trigger_mode", "absolute")
        self.anchor_offset = int(config.get("anchor_offset", -2))
        self.z_score_threshold = float(config.get("z_score_threshold", 2.0))
        self.ema_alpha = float(config.get("ema_alpha", 0.01))
        
        # GPU EMA Variables
        self.ema_mean_pos = torch.zeros(num_envs, device=device)
        self.ema_var_pos = torch.ones(num_envs, device=device)
        self.ema_mean_neg = torch.zeros(num_envs, device=device)
        self.ema_var_neg = torch.ones(num_envs, device=device)
        
        # Hashing config
        self.hash_bits = int(config.get("hash_bits", 12))
        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32, device=device)
        self.hash_proj = proj
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.long, device=device)
        self.hash_bit_values = bit_values
        
        # Hash memory as a GPU Tensor
        self.num_buckets = 2 ** self.hash_bits
        self.hash_memory = torch.zeros((self.num_buckets, self.max_bucket_size, 2), dtype=torch.long, device=device)
        self.bucket_lens = torch.zeros(self.num_buckets, dtype=torch.long, device=device)
        self.bucket_replace_idx = torch.zeros(self.num_buckets, dtype=torch.long, device=device)
        
        # Index to bucket mapping tensor
        # We preallocate enough for standard buffer lengths. It wraps around anyway.
        self.max_buf_len = int(config.get("max_buf_len", 2000000))
        self.index_to_bucket = torch.full((self.max_buf_len, num_envs), -1, dtype=torch.long, device=device)
        
        self.prev_v = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_keys = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        
        # GPU Ring Buffer for Active Anchors
        self.anchor_queue_capacity = int(config.get("anchor_queue_capacity", 8192))
        self.active_anchors = torch.zeros((self.anchor_queue_capacity, 4), dtype=torch.long, device=device)
        self.anchor_head = torch.zeros(1, dtype=torch.long, device=device)
        self.anchor_tail = torch.zeros(1, dtype=torch.long, device=device)
        self.anchor_count = torch.zeros(1, dtype=torch.long, device=device)

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        keys = (bits.to(torch.long) * self.hash_bit_values).sum(dim=-1)
        return keys

    def add_batch_transitions(self, v_t, base_indexes, base_envs, max_buf_len, skip_len=8):
        if not self.enabled:
            return 0, 0
            
        v_t_eval = v_t[:, skip_len:]
        delta_v_raw = v_t_eval[:, 1:] - v_t_eval[:, :-1]
        
        valid_mask_1d = torch.from_numpy(base_envs != -1).to(delta_v_raw.device)
        if not valid_mask_1d.any():
            return 0, 0
            
        valid_mask_2d = valid_mask_1d.unsqueeze(1)
        env_indices_full = torch.from_numpy(base_envs).to(delta_v_raw.device)
        
        pos_mask = (delta_v_raw > 0) & valid_mask_2d
        neg_mask = (delta_v_raw < 0) & valid_mask_2d
        
        if getattr(self, "trigger_mode", "absolute") == "z_score":
            if pos_mask.any():
                valid_delta_v_pos = delta_v_raw[pos_mask]
                env_indices_pos = env_indices_full.unsqueeze(1).expand_as(delta_v_raw)[pos_mask]
                
                counts = torch.bincount(env_indices_pos, minlength=self.num_envs).float()
                sums = torch.bincount(env_indices_pos, weights=valid_delta_v_pos, minlength=self.num_envs)
                env_means = torch.where(counts > 0, sums / counts, self.ema_mean_pos)
                
                sq_sums = torch.bincount(env_indices_pos, weights=valid_delta_v_pos**2, minlength=self.num_envs)
                env_vars = torch.where(counts > 0, (sq_sums / counts) - env_means**2, torch.zeros_like(self.ema_var_pos))
                
                update_mask = counts > 0
                diff = env_means - self.ema_mean_pos
                self.ema_mean_pos = torch.where(update_mask, self.ema_mean_pos + self.ema_alpha * diff, self.ema_mean_pos)
                new_var = (1 - self.ema_alpha) * (self.ema_var_pos + self.ema_alpha * (env_vars + diff**2))
                self.ema_var_pos = torch.where(update_mask, new_var, self.ema_var_pos)
                
                b_ema_mean_pos = self.ema_mean_pos[env_indices_pos]
                b_ema_var_pos = self.ema_var_pos[env_indices_pos]
                z_scores_pos = torch.abs(valid_delta_v_pos - b_ema_mean_pos) / (torch.sqrt(b_ema_var_pos) + 1e-8)
                triggered_pos = z_scores_pos >= self.z_score_threshold
            else:
                triggered_pos = torch.zeros(0, dtype=torch.bool, device=delta_v_raw.device)
                
            if neg_mask.any():
                valid_delta_v_neg = torch.abs(delta_v_raw[neg_mask])
                env_indices_neg = env_indices_full.unsqueeze(1).expand_as(delta_v_raw)[neg_mask]
                
                counts = torch.bincount(env_indices_neg, minlength=self.num_envs).float()
                sums = torch.bincount(env_indices_neg, weights=valid_delta_v_neg, minlength=self.num_envs)
                env_means = torch.where(counts > 0, sums / counts, self.ema_mean_neg)
                
                sq_sums = torch.bincount(env_indices_neg, weights=valid_delta_v_neg**2, minlength=self.num_envs)
                env_vars = torch.where(counts > 0, (sq_sums / counts) - env_means**2, torch.zeros_like(self.ema_var_neg))
                
                update_mask = counts > 0
                diff = env_means - self.ema_mean_neg
                self.ema_mean_neg = torch.where(update_mask, self.ema_mean_neg + self.ema_alpha * diff, self.ema_mean_neg)
                new_var = (1 - self.ema_alpha) * (self.ema_var_neg + self.ema_alpha * (env_vars + diff**2))
                self.ema_var_neg = torch.where(update_mask, new_var, self.ema_var_neg)
                
                b_ema_mean_neg = self.ema_mean_neg[env_indices_neg]
                b_ema_var_neg = self.ema_var_neg[env_indices_neg]
                z_scores_neg = torch.abs(valid_delta_v_neg - b_ema_mean_neg) / (torch.sqrt(b_ema_var_neg) + 1e-8)
                triggered_neg = z_scores_neg >= self.z_score_threshold
            else:
                triggered_neg = torch.zeros(0, dtype=torch.bool, device=delta_v_raw.device)
                
        else:
            triggered_pos = delta_v_raw[pos_mask] >= self.threshold if pos_mask.any() else torch.zeros(0, dtype=torch.bool, device=delta_v_raw.device)
            triggered_neg = torch.abs(delta_v_raw[neg_mask]) >= self.threshold if neg_mask.any() else torch.zeros(0, dtype=torch.bool, device=delta_v_raw.device)

        def process_triggers(mask, triggered_subset, sign):
            if not (mask.any() and triggered_subset.any()):
                return 0
                
            mask_indices = mask.nonzero(as_tuple=False)
            triggered_full_indices = mask_indices[triggered_subset]
            
            orig_b_idx = triggered_full_indices[:, 0]
            t_idx = triggered_full_indices[:, 1]
            
            env_idx = env_indices_full[orig_b_idx]
            base_ptr = torch.from_numpy(base_indexes).to(self.device)[orig_b_idx]
            
            anchor_ptr = (base_ptr + skip_len + 1 + t_idx + self.anchor_offset) % max_buf_len
            anchor_key = self.index_to_bucket[anchor_ptr, env_idx]
            
            valid = anchor_key != -1
            if not valid.any():
                return 0
                
            anchor_ptr = anchor_ptr[valid]
            env_idx = env_idx[valid]
            anchor_key = anchor_key[valid]
            signs = torch.full_like(anchor_ptr, sign)
            
            M = anchor_ptr.shape[0]
            anchors_to_push = torch.stack([anchor_ptr, env_idx, signs, anchor_key], dim=1)
            
            indices = (self.anchor_tail + torch.arange(M, device=self.device)) % self.anchor_queue_capacity
            self.active_anchors[indices] = anchors_to_push
            self.anchor_tail = (self.anchor_tail + M) % self.anchor_queue_capacity
            self.anchor_count = torch.clamp(self.anchor_count + M, max=self.anchor_queue_capacity)
            
            if self.anchor_count.item() == self.anchor_queue_capacity:
                self.anchor_head = self.anchor_tail

            return M
            
        num_triggered_pos = process_triggers(pos_mask, triggered_pos, sign=1)
        num_triggered_neg = process_triggers(neg_mask, triggered_neg, sign=-1)
        
        return num_triggered_pos, num_triggered_neg

    def add_transition(self, pointer, env_idx, latent_b):
        if not self.enabled:
            return 0
            
        keys = self._hash_keys(latent_b)
        
        if env_idx == -1:
            env_indices = torch.arange(self.num_envs, device=self.device)
            pointers = torch.full((self.num_envs,), pointer, dtype=torch.long, device=self.device)
        else:
            env_indices = torch.tensor([env_idx], device=self.device)
            pointers = torch.tensor([pointer], dtype=torch.long, device=self.device)
            keys = keys[env_idx:env_idx+1]
            
        self.prev_keys[env_indices] = keys
        self._insert_into_bucket(pointers, env_indices, keys)
        return 0

    def _insert_into_bucket(self, pointers, env_indices, keys):
        self.index_to_bucket[pointers % self.max_buf_len, env_indices] = keys
        
        M = pointers.shape[0]
        for i in range(M):
            p = pointers[i]
            e = env_indices[i]
            k = keys[i]
            
            b_size = self.bucket_lens[k]
            r_idx = self.bucket_replace_idx[k]
            
            if b_size < self.max_bucket_size:
                insert_idx = b_size
                self.bucket_lens[k] += 1
            else:
                insert_idx = r_idx
                self.bucket_replace_idx[k] = (r_idx + 1) % self.max_bucket_size
                
            self.hash_memory[k, insert_idx, 0] = p
            self.hash_memory[k, insert_idx, 1] = e

    def retrieve_contexts(self, replay_buffer, world_model, max_anchors, multiplier=5, target=5, max_contexts=256):
        if not self.enabled or self.anchor_count.item() == 0:
            return None, None, 0, 0.0, []
            
        pop_count = min(max_anchors, self.anchor_count.item())
        indices = (self.anchor_head + torch.arange(pop_count, device=self.device)) % self.anchor_queue_capacity
        popped_anchors = self.active_anchors[indices]
        self.anchor_head = (self.anchor_head + pop_count) % self.anchor_queue_capacity
        self.anchor_count -= pop_count
        
        anchor_ptrs = popped_anchors[:, 0]
        anchor_envs = popped_anchors[:, 1]
        anchor_keys = popped_anchors[:, 3]
        
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        
        all_sampled_ptrs = []
        all_sampled_envs = []
        all_is_exact = []
        all_anchor_idx = []
        
        for i in range(pop_count):
            a_ptr = anchor_ptrs[i]
            a_env = anchor_envs[i]
            a_key = anchor_keys[i]
            
            b_len = self.bucket_lens[a_key]
            target_k = multiplier * (target - 1)
            
            collected_ptrs = []
            collected_envs = []
            collected_is_exact = []
            
            exact_sample_k = min(target_k, b_len.item())
            if exact_sample_k > 0:
                rand_idx = torch.randperm(b_len.item(), device=self.device)[:exact_sample_k]
                collected_ptrs.append(self.hash_memory[a_key, rand_idx, 0])
                collected_envs.append(self.hash_memory[a_key, rand_idx, 1])
                collected_is_exact.append(torch.ones(exact_sample_k, dtype=torch.bool, device=self.device))
                

                        
            if len(collected_ptrs) == 0:
                continue
                
            sampled_ptrs = torch.cat(collected_ptrs)
            sampled_envs = torch.cat(collected_envs)
            is_exact = torch.cat(collected_is_exact)
            
            not_anchor = ~((sampled_ptrs == a_ptr) & (sampled_envs == a_env))
            sampled_ptrs = sampled_ptrs[not_anchor]
            sampled_envs = sampled_envs[not_anchor]
            is_exact = is_exact[not_anchor]
            
            valid_p = (sampled_ptrs >= 0) | (replay_buffer.length >= max_buf_len)
            valid_len = replay_buffer.length >= self.context_length
            valid = valid_p & valid_len
            
            sampled_ptrs = sampled_ptrs[valid]
            sampled_envs = sampled_envs[valid]
            is_exact = is_exact[valid]
            
            if sampled_ptrs.shape[0] == 0:
                continue
                
            all_sampled_ptrs.append(sampled_ptrs)
            all_sampled_envs.append(sampled_envs)
            all_is_exact.append(is_exact)
            all_anchor_idx.append(torch.full((sampled_ptrs.shape[0],), i, dtype=torch.long, device=self.device))
            
        if len(all_sampled_ptrs) == 0:
            return None, None, 0, 0.0, []
            
        flat_ptrs = torch.cat(all_sampled_ptrs)
        flat_envs = torch.cat(all_sampled_envs)
        flat_is_exact = torch.cat(all_is_exact)
        flat_anchor_idx = torch.cat(all_anchor_idx)
        
        curr_ptrs = flat_ptrs % max_buf_len
        
        if len(replay_buffer.termination_buffer.shape) == 2:
            obs_batch = replay_buffer.obs_buffer[curr_ptrs, flat_envs]
        else:
            obs_batch = replay_buffer.obs_buffer[curr_ptrs]
            
        obs_batch = obs_batch.float() / 255.0
        from einops import rearrange
        obs_batch = rearrange(obs_batch, "N H W C -> N 1 C H W")
        
        with torch.no_grad():
            encoded = world_model.encode_obs(obs_batch)
            encoded = encoded.squeeze(1)
            
        current_keys = self._hash_keys(encoded)
        
        required_keys = anchor_keys[flat_anchor_idx]
        hit_mask = (current_keys == required_keys)
        
        total_hit_rate = 0.0
        num_hit_rate_samples = 0
        
        for i in range(pop_count):
            anchor_is_exact = flat_is_exact[flat_anchor_idx == i]
            anchor_hits = hit_mask[flat_anchor_idx == i]
            
            if anchor_is_exact.shape[0] > 0 and anchor_is_exact.any():
                exact_hit_mask = anchor_hits[anchor_is_exact]
                if exact_hit_mask.shape[0] > 0:
                    hit_rate = exact_hit_mask.float().mean().item()
                    total_hit_rate += hit_rate
                    num_hit_rate_samples += 1
                    
        valid_anchor_idx = flat_anchor_idx[hit_mask]
        
        M = pop_count
        if valid_anchor_idx.shape[0] > 0:
            match_matrix = (valid_anchor_idx.unsqueeze(0) == torch.arange(M, device=self.device).unsqueeze(1))
            match_counts = match_matrix.cumsum(dim=1)
            keep_matrix = match_matrix & (match_counts <= target)
            keep_mask_1d = keep_matrix.any(dim=0)
            
            p_tensor = flat_ptrs[hit_mask][keep_mask_1d]
            env_tensor = flat_envs[hit_mask][keep_mask_1d]
        else:
            return None, None, 0, 0.0, []
            
        if p_tensor.shape[0] == 0:
            return None, None, 0, 0.0, []
            
        candidates_before_max = p_tensor.shape[0]
        
        if candidates_before_max > max_contexts:
            p_tensor = p_tensor[:max_contexts]
            env_tensor = env_tensor[:max_contexts]
            
        avg_hit_rate = total_hit_rate / max(1, num_hit_rate_samples)
        
        steps = torch.arange(self.context_length - 1, -1, -1, device=self.device)
        p_expanded = p_tensor.unsqueeze(1) - steps.unsqueeze(0)
        p_expanded = p_expanded % max_buf_len
        env_expanded = env_tensor.unsqueeze(1).expand(-1, self.context_length)
        
        if len(replay_buffer.termination_buffer.shape) == 2:
            terms = replay_buffer.termination_buffer[p_expanded, env_expanded]
        else:
            terms = replay_buffer.termination_buffer[p_expanded]
            
        invalid_mask = (steps.unsqueeze(0) > 0) & (terms > 0.5)
        is_invalid_seq = invalid_mask.any(dim=1)
        valid_seqs = ~is_invalid_seq
        
        p_expanded = p_expanded[valid_seqs]
        env_expanded = env_expanded[valid_seqs]
        
        if p_expanded.shape[0] == 0:
             return None, None, candidates_before_max, avg_hit_rate, []
             
        if len(replay_buffer.termination_buffer.shape) == 2:
            ret_obs = replay_buffer.obs_buffer[p_expanded, env_expanded]
            ret_action = replay_buffer.action_buffer[p_expanded, env_expanded]
        else:
            ret_obs = replay_buffer.obs_buffer[p_expanded]
            ret_action = replay_buffer.action_buffer[p_expanded]
            
        ret_obs = ret_obs.float() / 255.0
        ret_obs = rearrange(ret_obs, "B T H W C -> B T C H W")
        
        retrieved_indices = p_tensor[valid_seqs].cpu().tolist()
        
        return ret_obs, ret_action, candidates_before_max, avg_hit_rate, retrieved_indices

    @torch.no_grad()
    def rebuild_all_hash_buckets(self, replay_buffer, world_model, chunk_size=1024):
        self.bucket_lens.zero_()
        self.bucket_replace_idx.zero_()
        self.index_to_bucket.fill_(-1)
        
        if replay_buffer.length < self.context_length:
            return
            
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        valid_len = min(replay_buffer.length, max_buf_len)
        
        for start_idx in range(0, valid_len, chunk_size):
            end_idx = min(start_idx + chunk_size, valid_len)
            
            p_indices = torch.arange(start_idx, end_idx, device=self.device)
            env_indices = torch.arange(replay_buffer.num_envs, device=self.device)
            
            p_grid, env_grid = torch.meshgrid(p_indices, env_indices, indexing='ij')
            
            if len(replay_buffer.termination_buffer.shape) == 2:
                obs_chunk = replay_buffer.obs_buffer[p_grid, env_grid]
            else:
                obs_chunk = replay_buffer.obs_buffer[p_grid]
                
            obs_chunk = obs_chunk.reshape(-1, *obs_chunk.shape[2:]).float() / 255.0
            from einops import rearrange
            obs_chunk = rearrange(obs_chunk, "N H W C -> N 1 C H W")
            
            encoded = world_model.encode_obs(obs_chunk).squeeze(1)
            keys = self._hash_keys(encoded)
            
            flat_p = p_grid.reshape(-1)
            flat_env = env_grid.reshape(-1)
            
            self._insert_into_bucket(flat_p, flat_env, keys)
            
        print(f"[Retrieval] Global Rebuild completed. Re-hashed {valid_len * replay_buffer.num_envs} frames.")
