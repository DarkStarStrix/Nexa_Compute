# Nexa Shell Script Templates

The original `.sh` helpers are intentionally ignored because they often contain cluster‑specific credentials or hostnames. This directory provides sanitized templates that can be copied and customized locally.

## Usage

```bash
cp scripts/templates/bootstrap_node.template.sh bootstrap_node.sh
chmod +x bootstrap_node.sh
```

Then edit the copied file to fill in the environment variables marked as `CHANGEME` or export them before execution. The templates included:

| File | Purpose |
| --- | --- |
| `bootstrap_node.template.sh` | Rsyncs the repo and installs dependencies on a remote node over SSH. |
| `run_training.template.sh` | Launches single‑GPU or multi‑GPU training via `torchrun`. |
| `start_forge_services.template.sh` | Boots the API server and dashboard locally with sanitized logging. |

Feel free to add more templates beside these files. Keep secrets out of version control by copying the template and adding the resulting `.sh` file to your personal `.git/info/exclude` or relying on the default ignore rule.***

