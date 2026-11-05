# GitHub publication and CI

The checked-in project is ready for GitHub, but having a local Git commit or ZIP **does not mean anything was uploaded**. An authorized GitHub connection must be able to create a repository or publish to an approved existing repository. Do not place this standalone lab into an unrelated repository.

## Publish from a machine with GitHub CLI access

1. Install Git and [GitHub CLI](https://cli.github.com/), then authenticate with `gh auth login`.
2. Extract the source archive or clone the accompanying Git bundle (`git clone -b main ../weightspace-research-lab-v0.3.bundle weightspace-research-lab`).
3. From the lab folder, run:

   ```bash
   bash scripts/publish_to_github.sh sahiixx/weightspace-research-lab --private
   ```

This will create a **private, empty** repository if it does not already exist, then push `main`. For an intentionally public research release use `--public`. The script refuses to overwrite any repository with an existing branch, refuses to repoint `origin`, and avoids changing the current repository history.

If the repository already exists and is empty, the script pushes there. If it contains unrelated work, create a different empty repo instead. After publishing, use GitHub's **Actions** tab to inspect `weight-space-lab CI` results. On successful or failed CI runs the workflow attempts to upload JSON/Markdown audit evidence.

## What CI proves

- Tests, syntax compilation and synthetic experiments can execute on GitHub-hosted CPU runners.
- Reproducible 12-seed sweep, toy-model patch verification, and rejection controls run automatically.
- Report artifacts are viewable for 14 days.

CI does *not* download Qwen model weights or replicate MATH-500, GSM8K or IFEval research scores. Reproducing those claims requires the correct models, evaluation harness, datasets, parameter versions and a capable GPU environment.

## Recommended GitHub repository settings

Keep the initial project private until you have reviewed the research documentation and third-party license references. Enable branch protection or a ruleset requiring the `verify` job, and review workflow permission scopes before allowing contributions. `ci.yml` requests only `contents: read`. The lab has no credentials embedded in source code.

## Troubleshooting

- `gh: command not found`: install GitHub CLI, or create an empty repository through github.com and run `git remote add origin https://github.com/OWNER/REPO.git && git push -u origin main`.
- `git push` authentication failure: run `gh auth login` and `gh auth setup-git`, then retry.
- GitHub Actions pending: verify that Actions is enabled for the destination and check its first workflow run rather than assuming CI passed.
- PyTorch installation takes longer on a new runner: the workflow explicitly installs the CPU wheel and is not GPU dependent.
