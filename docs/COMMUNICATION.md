# Communication & Contribution Guidelines

This repository manages highly experimental Artificial Intelligence pipelines. Because we are dealing with high-dimensional matrices, non-deterministic training runs, and hardware-accelerated code, clear communication is critical.

## Pull Requests
1. **Branch Naming:** Branches must follow the format `feature/your-feature`, `bugfix/issue-description`, or `research/experiment-name`.
2. **Deterministic Verification:** Before submitting a PR, ensure `verify_mjx_pipeline.py` passes on your local machine using a fixed random seed.
3. **Training Deltas:** If your PR modifies reward structures or PPO hyperparameters, you must include a summary of the training convergence speed (e.g. "Achieved standing balance in 20M steps vs baseline 35M steps").

## Research Discussions
For proposing new neural architectures (e.g. Transformers over MLPs) or reward engineering strategies, open a GitHub Issue tagged with `[Research]`. Include mathematical justifications or links to relevant academic papers.
