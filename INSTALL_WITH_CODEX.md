# Deploy SACFactory locally with Codex

Use Codex to download, install and start SACFactory on your own computer. The
result is a local Streamlit application, normally at `http://127.0.0.1:8517`.
GitHub stores the source; it does not run this local service for you.

## Choose an installation route

You do not need to write code for routes A or B. Codex is the installation
assistant: it checks the computer, runs the setup commands and gives you a local
browser address. If a missing tool requires your approval, it should explain
what to install and guide you through that step.

| Route | Steps | Programming needed? |
| --- | --- | --- |
| **A. Codex downloads and installs** | Open an empty local folder in Codex; paste the short request below | No |
| **B. Download ZIP and use Codex** | Download and extract the GitHub ZIP; open the extracted project in Codex; request setup | No |
| **C. Manual terminal setup** | Use the platform-specific commands later in this guide | Terminal familiarity |

### Route A: the shortest request

Open Codex, choose a local folder for this project, and send:

```text
I have no programming experience. Please install and run
https://github.com/OscarLan-Fei/SACFactory on this computer.
Read its INSTALL_WITH_CODEX.md and follow the local deployment instructions.
Handle the setup commands for me, explain any prerequisite that needs my help,
and give me the browser address and a simple way to open it next time.
Preserve existing data and use DEMO for the initial check. Do not train models
or upload any local records during installation.
```

### Route B: download first

1. Open the [SACFactory repository](https://github.com/OscarLan-Fei/SACFactory).
2. Use **Code → Download ZIP**, then extract the archive to a writable local folder.
3. In Codex, open the extracted project folder containing `app.py` and this guide.
4. Send: **"Install and run this existing SACFactory folder locally. Follow
   INSTALL_WITH_CODEX.md. I do not have programming experience; handle the commands
   and explain how to start it next time. Preserve data and do not start training."**

A ZIP download does not need Git for the download step; dependency setup still
needs Python and internet access. Codex should skip cloning an existing extracted
project. The remaining sections give a detailed request, verification checklist
and the existing manual installation routes.

## 1. Open a local workspace

Start Codex with access to a local folder where you want the software installed.
Use a local task when the goal is to run on this computer. If the repository is
already downloaded, open its folder; otherwise, open an empty parent folder.

For current Codex setup and local-project instructions, see the
[official OpenAI quickstart](https://learn.chatgpt.com/docs/quickstart) and
[desktop documentation](https://learn.chatgpt.com/docs/app).
Codex access is configured separately from SACFactory. No language-model API key
is required for SACFactory's experiment management or numerical models.

## 2. Paste this request into Codex

```text
Install and run SACFactory locally on this computer from:
https://github.com/OscarLan-Fei/SACFactory

1. Inspect the workspace and operating system. If the repository is absent,
   clone it into a SACFactory subfolder. If it exists, inspect its state and
   preserve local changes, records and checkpoints; do not reset it.
2. Read README.md, INSTALL_WITH_CODEX.md and requirements.txt. Locate Git and
   Python 3.12. If a prerequisite is missing, explain the required installation
   and follow this environment's approval process before changing the system.
3. Create a repository-local .venv. Install requirements.txt using its Python,
   run pip check, and verify imports of Streamlit, NumPy, pandas, scikit-learn
   and PyTorch. Do not install into system Python or change the model code to
   bypass dependency errors.
4. Start app.py using Streamlit bound to 127.0.0.1, normally on port 8517. Check
   for an existing instance first. If another application occupies that port,
   choose a free local port without stopping it. On Windows, use a hidden
   background process if a persistent launcher is needed.
5. Check /_stcore/health at the selected URL. Open the application if browser
   tools are available; otherwise give me the URL. Verify that DEMO loads
   without using real experiments. Do not start training as an installation test.
6. Report the repository path, environment path, URL and exact start, restart
   and stop commands. Explain how to open the application next time.

Keep experimental records, credentials and trained weights local. Do not push
files to GitHub, expose the server to the network, change firewall settings or
delete records. CPU execution is sufficient for the first launch; inspect GPU
support separately if I later request it.
```

Codex may request environment-specific permissions. Review the requested action;
this prompt does not bypass the approval rules of your installation.

## 3. Check the result

| Check | Expected result |
| --- | --- |
| Folder | Contains `app.py`, `requirements.txt` and `sacfactory/` |
| Environment | A repository-local `.venv` |
| Dependencies | `pip check` reports no broken requirements |
| Service | `/_stcore/health` responds successfully on the chosen loopback port |
| Browser | SACFactory opens with REAL and DEMO data spaces |
| Initial data | DEMO loads; REAL contains no bundled private experiments |
| Model | Bundled pretrained models work in **Discovery lab → Predict** without training; new local deployments are optional |

Use **Python 3.12**. Internet access is needed to download source and dependencies.
First installation can take longer than later starts. CPU execution is supported.
NVIDIA GPU execution requires a compatible driver and CUDA-enabled PyTorch build;
package installation alone does not establish that CUDA is available.

## Manual commands: Windows PowerShell

Run from the parent directory where the repository should be created. Skip the
clone and directory-change steps if you already opened the repository folder.

```powershell
git clone https://github.com/OscarLan-Fei/SACFactory.git
Set-Location SACFactory
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import streamlit, numpy, pandas, sklearn, torch; print('Imports OK')"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517
```

Open `http://127.0.0.1:8517`. Keep the terminal running; **Ctrl+C** stops this
foreground service. Explicit interpreter paths avoid requiring virtual-environment
activation or a change to PowerShell's global execution policy.

After dependencies are installed, you can also double-click
**Start-SACFactory.cmd** to start or restart the service and open the default URL.
For a launcher-managed process, the repository provides `Stop-SACFactory.ps1`;
ask Codex to run it from this installation. Stop a foreground process with Ctrl+C.

## Manual commands: macOS / Linux

Install Python 3.12 and Git first, then run:

```bash
git clone https://github.com/OscarLan-Fei/SACFactory.git
cd SACFactory
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -c "import streamlit, numpy, pandas, sklearn, torch; print('Imports OK')"
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517
```

Open the local URL and keep the terminal running. **Ctrl+C** stops the service.
The bundled `.cmd` and `.ps1` launchers are for Windows.

## Open it again later

The page works while the local service is running. Closing a browser tab does
not necessarily stop it; restarting the computer does. On Windows, double-click
**Start-SACFactory.cmd**. On any platform, rerun the final Streamlit command from
the repository folder using its existing `.venv`. You need not reinstall on each
launch. Alternatively, paste this into Codex:

```text
Start the existing SACFactory installation in this workspace using its .venv.
Preserve records and checkpoints. Check the existing service and port first,
then verify the health endpoint and give me the URL. Do not reinstall or retrain
unless an actual problem requires it.
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Connection refused | Start the service; check the actual port and health response |
| Python command not found | Ask Codex to locate Python 3.12 or help install that prerequisite |
| `ModuleNotFoundError` | Use the repository's `.venv` interpreter and install requirements there |
| Dependency resolution error | Keep the full error; check Python version and package builds before changing constraints |
| Port occupied | Use another free port, such as 8532, and open that URL |
| CUDA unavailable | Use CPU; inspect `torch.cuda.is_available()` and driver/build compatibility separately |
| Immediate prediction | Use **Discovery lab → Predict** with a bundled pretrained model |
| No local Next batch model | Validate data, then click **Train deployment model**; the local deployment is separate from the public inference release |
| Fit uses fewer rows | Inspect calculation scope and eligibility; missing measurements are not zero |
| Research conversation fails | Check **Language model**; numerical modeling does not require that API |

For updates, ask Codex to inspect local changes and fetch upstream before
updating. Preserve `data/` and checkpoints rather than replacing the installation.
Continue with the [User guide](USER_GUIDE.md).
