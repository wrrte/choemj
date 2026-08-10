import torch
import random
import numpy as np
from collections import deque

class FastHashBucket:
    """O(N) 병목을 제거하고 완전한 O(1) 연산을 지원하는 딕셔너리 기반 해시 버킷"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = [] # 랜덤 샘플링을 위한 리스트 (index 저장)
        self.data_map = {} # dict mapping index_item -> (list_index)

    def add_or_update(self, index):
        """데이터를 추가하거나 업데이트하며, 용량 초과로 삭제된 인덱스가 있다면 반환합니다."""
        if index in self.data_map:
            return None
        
        if len(self.items) < self.max_size:
            list_idx = len(self.items)
            self.items.append(index)
            self.data_map[index] = list_idx
            return None
        else:
            # Replace randomly
            replace_list_idx = random.randrange(self.max_size)
            old_index = self.items[replace_list_idx]
            del self.data_map[old_index]
            
            self.items[replace_list_idx] = index
            self.data_map[index] = replace_list_idx
            return old_index

    def remove(self, index):
        """특정 인덱스를 O(1) 시간에 삭제합니다."""
        if index not in self.data_map:
            return
        list_idx = self.data_map[index]
        last_index = self.items[-1]
        
        # swap with last
        self.items[list_idx] = last_index
        self.data_map[last_index] = list_idx
        
        self.items.pop()
        del self.data_map[index]

    def sample(self, k, exclude=None):
        """무작위로 최대 k개를 샘플링하되 제외할 인덱스가 있으면 제외"""
        if not self.items:
            return []
        
        pool = self.items
        if exclude is not None and exclude in self.data_map:
            # exclude가 리스트에 있다면 제외하고 샘플링 (스왑 기법 등을 쓸 수 있지만 단순 복사/필터링 사용)
            pool = [x for x in self.items if x != exclude]
            
        k = min(k, len(pool))
        if k == 0:
            return []
            
        indices = random.sample(pool, k)
        return indices

    def __len__(self):
        return len(self.items)


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
        
        self.ema_mean = np.zeros(num_envs)
        self.ema_var = np.ones(num_envs)
        
        # Hashing config
        self.hash_bits = int(config.get("hash_bits", 12))
        self.hash_sample_mode = config.get("hash_sample_mode", "probs")
        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32, device=device)
        self.hash_proj = proj
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64, device=device)
        self.hash_bit_values = bit_values
        
        self.hash_memory = {} # key -> FastHashBucket
        
        # Mapping (pointer, env_idx) -> hash_key (to remove old keys when overwritten)
        self.index_to_bucket = {} 
        
        self.prev_v = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_keys = [-1] * num_envs
        
        self.active_anchors = deque()

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        keys = (bits.to(torch.int64) * self.hash_bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def add_batch_transitions(self, v_t, reward, termination, gamma, base_indexes, base_envs, max_buf_len, skip_len=8, is_warmup=False):
        if not self.enabled:
            return 0, 0
            
        v_t_eval = v_t[:, skip_len:]
        reward_eval = reward[:, skip_len:].squeeze(-1) if reward.dim() == 3 else reward[:, skip_len:]
        term_eval = termination[:, skip_len:].squeeze(-1) if termination.dim() == 3 else termination[:, skip_len:]
        
        v_curr = v_t_eval[:, :-1]
        v_next = v_t_eval[:, 1:]
        r_curr = reward_eval[:, :-1]
        d_curr = term_eval[:, :-1]
        
        target_v = r_curr + gamma * v_next * (1.0 - d_curr)
        delta_v_raw = target_v - v_curr
        
        valid_mask_1d = torch.from_numpy(base_envs != -1).to(delta_v_raw.device)
        if not valid_mask_1d.any():
            return 0, 0
            
        valid_mask_2d = valid_mask_1d.unsqueeze(1)
        env_indices_full = torch.from_numpy(base_envs).to(delta_v_raw.device)
        
        abs_delta_v = torch.abs(delta_v_raw)
        
        metric = torch.full_like(delta_v_raw, float('-inf'))
        
        if getattr(self, "trigger_mode", "absolute") == "z_score":
            ema_mean_t = torch.from_numpy(self.ema_mean).to(delta_v_raw.device, dtype=torch.float32)
            ema_var_t = torch.from_numpy(self.ema_var).to(delta_v_raw.device, dtype=torch.float32)
            
            b_ema_mean = ema_mean_t[env_indices_full].unsqueeze(1).expand_as(delta_v_raw)
            b_ema_var = ema_var_t[env_indices_full].unsqueeze(1).expand_as(delta_v_raw)
            
            z_scores_full = torch.abs(abs_delta_v - b_ema_mean) / (torch.sqrt(b_ema_var) + 1e-8)
            metric[valid_mask_2d] = z_scores_full[valid_mask_2d]
            
            if valid_mask_2d.any():
                valid_abs_delta_v = abs_delta_v[valid_mask_2d]
                env_indices_valid = env_indices_full.unsqueeze(1).expand_as(delta_v_raw)[valid_mask_2d]
                
                for i in range(self.num_envs):
                    env_mask = (env_indices_valid == i)
                    if env_mask.any():
                        env_delta_v = valid_abs_delta_v[env_mask]
                        env_mean = env_delta_v.mean().item()
                        env_var = env_delta_v.var(unbiased=False).item() if env_delta_v.numel() > 1 else 0.0
                        
                        diff = env_mean - self.ema_mean[i]
                        self.ema_mean[i] += self.ema_alpha * diff
                        self.ema_var[i] = (1 - self.ema_alpha) * (self.ema_var[i] + self.ema_alpha * (env_var + diff**2))
            
            threshold = self.z_score_threshold
        else:
            metric[valid_mask_2d] = abs_delta_v[valid_mask_2d]
            threshold = self.threshold

        max_val, max_idx = metric.max(dim=1)
        triggered_b = max_val >= threshold
        
        def process_max_triggers(triggered_b, max_idx):
            num_trig = 0
            if triggered_b.any():
                b_indices = triggered_b.nonzero(as_tuple=True)[0]
                t_indices = max_idx[b_indices]
                
                for i in range(len(b_indices)):
                    orig_b_idx = b_indices[i].item()
                    t_idx = t_indices[i].item()
                    
                    env_idx = base_envs[orig_b_idx]
                    base_ptr = base_indexes[orig_b_idx]
                    
                    anchor_ptr = int(base_ptr + skip_len + 1 + t_idx + self.anchor_offset) % max_buf_len
                    
                    anchor_key = self.index_to_bucket.get((anchor_ptr, env_idx), -1)
                    if anchor_key != -1:
                        anchor = (anchor_ptr, env_idx)
                        self.active_anchors.append((anchor, anchor_key))
                        num_trig += 1
            return num_trig
            
        if is_warmup:
            return 0
            
        num_triggered = process_max_triggers(triggered_b, max_idx)
        
        return num_triggered

    def add_transition(self, pointer, env_idx, latent_b):
        """
        pointer: the buffer pointer at which current_obs was just appended.
        latent_b: (num_envs, latent_dim)
        """
        if not self.enabled:
            return 0
            
        keys = self._hash_keys(latent_b)
        
        for i in range(self.num_envs):
            if env_idx != -1 and i != env_idx:
                continue # for individual updates if needed, though usually batch
                
            self.prev_keys[i] = keys[i]
            
            # Add current latent to hash bucket
            self._insert_into_bucket(pointer, i, keys[i])
            
        return 0

    def _insert_into_bucket(self, pointer, env_idx, key):
        idx_tuple = (pointer, env_idx)
        old_key = self.index_to_bucket.get(idx_tuple, -1)
        
        if old_key != -1 and old_key != key:
            old_queue = self.hash_memory.get(old_key)
            if old_queue is not None:
                old_queue.remove(idx_tuple)
                
        queue = self.hash_memory.get(key)
        if queue is None:
            queue = FastHashBucket(max_size=self.max_bucket_size)
            self.hash_memory[key] = queue
            
        evicted_idx = queue.add_or_update(idx_tuple)
        if evicted_idx is not None:
            if evicted_idx in self.index_to_bucket:
                del self.index_to_bucket[evicted_idx]
                
        self.index_to_bucket[idx_tuple] = key

    def retrieve_contexts(self, replay_buffer, world_model, max_anchors, multiplier=5, target=5, max_contexts=256):
        """
        Pops up to `max_anchors` from `active_anchors` and retrieves up to max contexts in total.
        Implements lazy recomputation using single frame encoding.
        """
        if not self.enabled or len(self.active_anchors) == 0:
            return None, None, 0, 0.0, [], 0
            
        popped_anchors = []
        for _ in range(min(max_anchors, len(self.active_anchors))):
            popped_anchors.append(self.active_anchors.popleft())
            
        retrieved_obs_list = []
        retrieved_action_list = []
        
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        
        anchor_groups = []
        
        total_hit_rate = 0.0
        num_hit_rate_samples = 0
        
        for anchor_tuple, anchor_key in popped_anchors:
            # anchor_tuple is (anchor_ptr, env_idx)
            anchor_ptr, anchor_env_idx = anchor_tuple
            queue = self.hash_memory.get(anchor_key)
            if not queue:
                continue
                
            sampled_indices = queue.sample(k=multiplier*(target-1), exclude=(anchor_ptr, anchor_env_idx))
                
            obs_list = []
            valid_sampled = []
            for (p, env_idx) in sampled_indices:
                curr_p = p % max_buf_len
                if not replay_buffer.store_on_gpu and p < 0 and replay_buffer.length < max_buf_len:
                    continue
                if replay_buffer.length < self.context_length:
                    continue
                obs_list.append(replay_buffer.obs_buffer[curr_p, env_idx:env_idx+1])
                valid_sampled.append((p, env_idx))
                
            if obs_list:
                if replay_buffer.store_on_gpu:
                    obs_tensor = torch.cat(obs_list, dim=0).float() / 255.0
                else:
                    import numpy as np
                    obs_arr = np.concatenate(obs_list, axis=0)
                    obs_tensor = torch.from_numpy(obs_arr).float().cuda() / 255.0
                    
                from einops import rearrange
                obs_tensor = rearrange(obs_tensor, "N H W C -> N 1 C H W")
                
                with torch.no_grad():
                    encoded = world_model.encode_obs(obs_tensor, sample_mode=self.hash_sample_mode) # [N, 1, latent_dim]
                    encoded = encoded.squeeze(1) # [N, latent_dim]
                    
                current_keys = self._hash_keys(encoded)
                
                if len(current_keys) > 0:
                    hit_rate = (torch.tensor(current_keys) == anchor_key).float().mean().item()
                    total_hit_rate += hit_rate
                    num_hit_rate_samples += 1
            else:
                current_keys = []
            
            # Filter matches
            matched_indices = [(anchor_ptr, anchor_env_idx)] # 앵커를 항상 첫 번째 타겟으로 추가
            for i_c, k_c in enumerate(current_keys):
                if k_c == anchor_key:
                    matched_indices.append(valid_sampled[i_c])
                    if len(matched_indices) >= target:
                        break
                        
            if matched_indices:
                anchor_groups.append(matched_indices)
            
        candidates_before_max = sum(len(g) for g in anchor_groups)
        
        final_anchor_groups = []
        total_candidates = 0
        for group in anchor_groups:
            if total_candidates >= max_contexts:
                break
            space = max_contexts - total_candidates
            to_add = group[:space]
            total_candidates += len(to_add)
            final_anchor_groups.append(to_add)
            
        retrieved_obs_list = []
        retrieved_action_list = []
        retrieved_weights = []
        valid_anchors_count = 0
        
        for group in final_anchor_groups:
            valid_group_obs = []
            valid_group_action = []
            
            for (p, env_idx) in group:
                valid = True
                obs_chunk = []
                action_chunk = []
                for step in range(self.context_length - 1, -1, -1):
                    curr_p = (p - step) % max_buf_len
                    term = replay_buffer.termination_buffer[curr_p, env_idx]
                    if step > 0 and term > 0.5:
                        valid = False
                        break
                        
                if valid:
                    for step in range(self.context_length - 1, -1, -1):
                        curr_p = (p - step) % max_buf_len
                        obs_chunk.append(replay_buffer.obs_buffer[curr_p, env_idx:env_idx+1])
                        action_chunk.append(replay_buffer.action_buffer[curr_p, env_idx:env_idx+1])
                        
                    if replay_buffer.store_on_gpu:
                        obs_tensor = torch.stack(obs_chunk, dim=0)
                        action_tensor = torch.stack(action_chunk, dim=0)
                    else:
                        import numpy as np
                        obs_tensor = np.stack(obs_chunk, axis=0)
                        action_tensor = np.stack(action_chunk, axis=0)
                        
                    valid_group_obs.append(obs_tensor)
                    valid_group_action.append(action_tensor)
                
            if valid_group_obs:
                valid_anchors_count += 1
                group_weight = 1.0 / len(valid_group_obs)
                for _ in range(len(valid_group_obs)):
                    retrieved_weights.append(group_weight)
                
                retrieved_obs_list.extend(valid_group_obs)
                retrieved_action_list.extend(valid_group_action)
                
        avg_hit_rate = total_hit_rate / max(1, num_hit_rate_samples)
        if len(retrieved_obs_list) == 0:
            return None, None, candidates_before_max, avg_hit_rate, [], 0
            
        if replay_buffer.store_on_gpu:
            ret_obs = torch.cat(retrieved_obs_list, dim=1).float() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = torch.cat(retrieved_action_list, dim=1).transpose(0, 1) # [B, T]
        else:
            import numpy as np
            ret_obs = np.concatenate(retrieved_obs_list, axis=1)
            ret_obs = torch.from_numpy(ret_obs).float().cuda() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = np.concatenate(retrieved_action_list, axis=1)
            ret_action = torch.from_numpy(ret_action).cuda().transpose(0, 1) # [B, T]
            
        return ret_obs, ret_action, candidates_before_max, avg_hit_rate, retrieved_weights, valid_anchors_count

    @torch.no_grad()
    def rebuild_all_hash_buckets(self, replay_buffer, world_model, chunk_size=1024):
        """
        Clears all hash buckets and re-hashes all valid transitions in the replay buffer 
        using the latest world model to fix representation drift (Global Rebuild).
        """
        self.hash_memory.clear()
        self.index_to_bucket.clear()
        
        if replay_buffer.length < self.context_length:
            return
            
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        valid_len = min(replay_buffer.length, max_buf_len)
        
        # We process in chunks to prevent GPU OOM
        for start_idx in range(0, valid_len, chunk_size):
            end_idx = min(start_idx + chunk_size, valid_len)
            current_chunk_size = end_idx - start_idx
            
            # Prepare observations for the chunk across all envs
            obs_list = []
            valid_indices = []
            
            for p in range(start_idx, end_idx):
                for env_idx in range(replay_buffer.num_envs):
                    # For simplicity, we just use the current frame without context window check here
                    # since we only need its latent for hashing. 
                    # If it's near termination, it might be an issue, but hashing the single frame is fine.
                    obs_list.append(replay_buffer.obs_buffer[p, env_idx:env_idx+1])
                    valid_indices.append((p, env_idx))
                    
            if len(obs_list) == 0:
                continue
                
            if replay_buffer.store_on_gpu:
                obs_tensor = torch.cat(obs_list, dim=0).float() / 255.0
            else:
                import numpy as np
                obs_arr = np.concatenate(obs_list, axis=0)
                obs_tensor = torch.from_numpy(obs_arr).float().cuda() / 255.0
                
            from einops import rearrange
            obs_tensor = rearrange(obs_tensor, "N H W C -> N 1 C H W")
            
            encoded = world_model.encode_obs(obs_tensor, sample_mode=self.hash_sample_mode)
            encoded = encoded.squeeze(1)
            keys = self._hash_keys(encoded)
            
            for (p, env_idx), key in zip(valid_indices, keys):
                self._insert_into_bucket(p, env_idx, key)
                
        print(f"[Retrieval] Global Rebuild completed. Re-hashed {valid_len * replay_buffer.num_envs} frames.")
