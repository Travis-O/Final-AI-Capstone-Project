from mistralai import Mistral
import json
from config import MISTRAL_API_KEY
import re

class ResearchGPTAssistant:
    def __init__(self, config, document_processor):
        self.config = config
        self.doc_processor = document_processor
        self.mistral_client = None
        try:
            if MISTRAL_API_KEY and MISTRAL_API_KEY != "WFDkKEBOwmsrKDKFKqzTqe5IYMffpnCX":
                self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        except Exception:
            self.mistral_client = None
        self.conversation_history = []
        self.prompts = self._load_prompt_templates()

    def _load_prompt_templates(self):
        return {
            "chain_of_thought": (
                "You are an expert research assistant.\n"
                "Use the provided context to reason step by step and then answer clearly.\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nThink step-by-step, then provide a concise answer."
            ),
            "self_consistency": (
                "Generate {n} diverse reasoning paths for the following question using the context. "
                "Return each as a short answer.\nContext:\n{context}\n\nQuestion: {query}"
            ),
            "react_research": (
                "Follow a ReAct loop: Thought -> Action -> Observation. Actions: Search, Analyze, Summarize.\n"
                "Context:\n{context}\n\nQuestion: {query}"
            ),
            "document_summary": (
                "Summarize the key findings, methods, and conclusions from the context.\nContext:\n{context}"
            ),
            "qa_with_context": (
                "Answer the question using only the context. If unknown, say so.\nContext:\n{context}\n\nQuestion: {query}"
            ),
            "verify_answer": (
                "Given the original question and context, verify the answer. "
                "Point out any missing or unsupported claims and provide an improved answer.\n"
                "Question: {query}\nContext:\n{context}\nAnswer:\n{answer}"
            ),
        }

    def _call_mistral(self, prompt, temperature=None, max_tokens=400):
        temperature = getattr(self.config, "TEMPERATURE", 0.7) if temperature is None else temperature
        if not self.mistral_client:
            return None
        try:
            messages = [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ]
            resp = self.mistral_client.chat.complete(
                model=getattr(self.config, "MODEL_NAME", "mistral-large-latest"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if hasattr(resp, "choices") and resp.choices:
                msg = resp.choices[0].message
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
            return str(resp)
        except Exception:
            return None

    def _build_context(self, chunks, limit=5):
        parts = []
        for ch in (chunks or [])[:limit]:
            if isinstance(ch, (list, tuple)) and ch:
                parts.append(str(ch[0]))
        return "\n\n".join(parts).strip()

    def chain_of_thought_reasoning(self, query, context_chunks):
        context = self._build_context(context_chunks)
        prompt = self.prompts["chain_of_thought"].format(context=context, query=query)
        out = self._call_mistral(prompt)
        return out if out else (context[:800] if context else "")

    def self_consistency_generate(self, query, context_chunks, num_attempts=3):
        context = self._build_context(context_chunks)
        prompt = self.prompts["self_consistency"].format(context=context, query=query, n=num_attempts)
        outputs = []
        for _ in range(max(1, num_attempts)):
            out = self._call_mistral(prompt)
            if not out:
                break
            outputs.append(out.strip())
        if not outputs:
            return self.chain_of_thought_reasoning(query, context_chunks)
        canon = [re.sub(r"\s+", " ", o.strip().lower()) for o in outputs]
        best = max(set(canon), key=canon.count)
        return outputs[canon.index(best)]

    def react_research_workflow(self, query, max_steps=5):
        steps = []
        context_chunks = self.doc_processor.find_similar_chunks(query, top_k=5)
        context = self._build_context(context_chunks)
        for i in range(max_steps):
            thought = f"Identify information gaps for: {query}"
            action = "Search"
            observation = context if context else "No relevant context found."
            steps.append({"step": i + 1, "thought": thought, "action": action, "observation": observation})
            if self._should_conclude_workflow(observation):
                break
        final_answer = self.chain_of_thought_reasoning(query, context_chunks)
        return {"workflow_steps": steps, "final_answer": final_answer}

    def _should_conclude_workflow(self, observation):
        if not observation:
            return False
        return len(observation) > 300

    def verify_and_edit_answer(self, answer, original_query, context):
        prompt = self.prompts["verify_answer"].format(query=original_query, context=context, answer=answer)
        out = self._call_mistral(prompt, temperature=0.2, max_tokens=300)
        if not out:
            return {"original_answer": answer, "improved_answer": answer}
        return {"original_answer": answer, "improved_answer": out}

    def answer_research_question(self, query, use_cot=True, use_verification=True):
        relevant_chunks = self.doc_processor.find_similar_chunks(query, top_k=5)
        if use_cot:
            answer = self.chain_of_thought_reasoning(query, relevant_chunks)
        else:
            context = self._build_context(relevant_chunks)
            prompt = self.prompts["qa_with_context"].format(context=context, query=query)
            out = self._call_mistral(prompt)
            answer = out if out else (context[:800] if context else "")
        verification_data = None
        if use_verification:
            verification_data = self.verify_and_edit_answer(answer, query, self._build_context(relevant_chunks))
            final_answer = verification_data["improved_answer"]
        else:
            final_answer = answer
        return {
            "query": query,
            "relevant_documents": len(relevant_chunks),
            "answer": final_answer,
            "verification": verification_data,
            "sources_used": [c[2] for c in (relevant_chunks or [])],
        }