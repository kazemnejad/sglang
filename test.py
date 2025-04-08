from pathlib import Path
import sglang as sgl
import os

os.environ["HF_HOME"] = str(Path.home() / "scratch" / "treetune_next" / "experiments" / "hf_home")

def main():
    llm = sgl.Engine(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        is_embedding=True,
    )
    o = llm.encode(["Hello, world!"])
    print(o)

if __name__ == "__main__":
    main()
