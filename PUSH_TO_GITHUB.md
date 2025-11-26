# Push to GitHub Instructions

## Current Status ✅

Your repository is ready to push! All files have been:
- ✅ Organized into proper directories (src/, logs/, results/, plots/, archive/)
- ✅ Committed to git with descriptive message
- ✅ Branch renamed to `main`
- ✅ Remote origin configured: https://github.com/cogito233/cs885_course_project.git

## Files Summary

**Total files committed**: 84 files
- Source code: `src/` directory
- Results: `results/` directory (JSON summaries)
- Logs: `logs/` directory
- Plots: `plots/` directory (visualizations)
- Archive: `archive/` directory (historical experiments)
- Documentation: README.md, USAGE.md, requirements.txt

**Files excluded** (via .gitignore):
- Large data files: `*.jsonl` (660MB)
- Python cache: `__pycache__/`
- Virtual environment: `.venv/`

## Next Steps - Push to GitHub

### Option 1: Using HTTPS (Recommended)

```bash
cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854

# Push to GitHub (will prompt for credentials)
git push -u origin main
```

You'll need to provide:
- **Username**: Your GitHub username
- **Password**: Your Personal Access Token (NOT your GitHub password)

#### How to create a Personal Access Token:
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (all)
4. Copy the token (save it securely!)
5. Use this token as your password when pushing

### Option 2: Using SSH

If you prefer SSH (no password prompts):

```bash
cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854

# Change remote to SSH
git remote set-url origin git@github.com:cogito233/cs885_course_project.git

# Push to GitHub
git push -u origin main
```

**Prerequisites**: You need SSH keys configured with GitHub
- Check: `ssh -T git@github.com`
- Setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Option 3: Manual Push (if remote issues)

```bash
cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854

# Verify current status
git log --oneline -1
git remote -v

# Try push with verbose output
GIT_CURL_VERBOSE=1 git push -u origin main
```

## After Successful Push

Once pushed, your repository will be available at:
**https://github.com/cogito233/cs885_course_project**

### Update README links

The plots images in README.md currently use local paths. After pushing, you may want to update them to use GitHub raw URLs:

```markdown
![GPU1 Comparison](https://raw.githubusercontent.com/cogito233/cs885_course_project/main/plots/gpu1/batch_size_comparison.png)
```

## Troubleshooting

### Error: "Authentication failed"
- Use Personal Access Token instead of password
- Verify token has `repo` permissions
- Try: `git config --global credential.helper cache`

### Error: "Repository not found"
- Verify repository exists at https://github.com/cogito233/cs885_course_project
- Check repository is not private (or you have access)

### Error: "failed to push some refs"
- If remote has commits you don't have:
  ```bash
  git pull origin main --rebase
  git push -u origin main
  ```

### Large file warnings
- Our .gitignore excludes .jsonl files
- If you get warnings about large files, verify:
  ```bash
  git ls-files | xargs -I {} ls -lh {} | sort -k5 -hr | head -20
  ```

## Verify Upload

After pushing, check on GitHub:
1. Go to https://github.com/cogito233/cs885_course_project
2. Verify file structure matches local
3. Check README.md renders correctly
4. View plots in plots/ directory

## Quick Reference

```bash
# Check what's ready to push
git status
git log --oneline -5

# View remote configuration
git remote -v

# Push to GitHub
git push -u origin main

# If successful, subsequent pushes:
git push
```

## Summary of Repository Structure

```
cs885_course_project/
├── README.md                 # Main documentation (English)
├── USAGE.md                  # Usage guide with examples
├── requirements.txt          # Python dependencies
├── RUN_COMMAND.sh           # Benchmark execution script
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── benchmark_per_turn.py          # Main benchmark (Stateful KV)
│   ├── plot_metrics.py                # Visualization
│   └── ...
│
├── results/                 # Test results (JSON summaries)
│   ├── per_turn_summary_gpu1_*.json
│   └── ...
│
├── logs/                    # Execution logs
│   ├── BENCHMARK_PER_TURN_GPU1_BS34.log
│   └── ...
│
├── plots/                   # Performance visualizations
│   ├── gpu1/
│   └── gpu2/
│
└── archive/                 # Historical experiments
    └── ...
```

## Need Help?

- GitHub Authentication: https://docs.github.com/en/authentication
- Git basics: https://git-scm.com/book/en/v2
- SSH setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

