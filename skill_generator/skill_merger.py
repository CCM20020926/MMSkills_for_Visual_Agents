from sentence_transformers import SentenceTransformer
import json


class SkillMerger:
    def __init__(self, llm_client, meta_guide):
        self.llm = llm_client
        self.guide = meta_guide
    
    def is_too_broad(self, skill):
        return False  # 暂未具体实现
    
    def merge(self, all_skills):
        # 嵌入并聚类
        enc = SentenceTransformer('all-MiniLM-L6-v2')
        names = [s['skill_name'] for s in all_skills]
        emb = enc.encode(names)
        
        # 使用HDBSCAN或DBSCAN进行密度聚类
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
        labels = clusterer.fit_predict(emb)
        
        merged = []
        for label in set(labels):
            if label == -1:
                merged.extend([all_skills[i] for i in range(len(labels)) if labels[i]==-1])
            else:
                group = [all_skills[i] for i in range(len(labels)) if labels[i]==label]
                merged_skill = self.merge_group(group)
                # 检查是否为过宽技能
                if not self.is_too_broad(merged_skill):
                    merged.append(merged_skill)
        return merged
    
    def merge_group(self, group):
        prompt = f"""
        Merge the following similar skill proposals into one generalized skill. 
        Use the most general description, combine workflow steps, unify completion criteria.
        Keep the skill_name concise.
        Skills: {json.dumps(group)}
        Output a single JSON object with keys: skill_name, description, workflow, completion_criteria, covered_task_ids (union).
        """
        
        return json.loads(self.llm.invoke(prompt))