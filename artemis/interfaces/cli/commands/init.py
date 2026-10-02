# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive setup wizard for first-time ARTEMIS onboarding (artemis init)."""

import shutil
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from artemis.config.paths import get_env_file


def init_command() -> None:
    """Interactive quickstart wizard to configure API keys and connect your device."""
    console = Console()
    console.print()
    console.print(
        Panel(
            "[bold cyan]☕ Welcome to Artemis Mobile Testing Framework[/bold cyan]\n"
            "[dim]Let's get you set up in 10 seconds![/dim]",
            expand=False,
        )
    )

    # 1. Select Provider
    console.print("\n[bold]Step 1: Choose Primary LLM Provider[/bold]")
    console.print("  [1] Claude Code (Recommended - local login or setup token)")
    console.print("  [2] Signed-in Codex CLI")
    console.print("  [3] Google Gemini")
    console.print("  [4] OpenAI API")
    console.print("  [5] Anthropic API")
    console.print("  [6] OpenRouter")

    choice = Prompt.ask("Select provider", choices=["1", "2", "3", "4", "5", "6"], default="1")
    provider_map = {
        "1": (None, "Claude Code", "claude-code", "claude-code"),
        "2": (None, "Codex CLI", "codex", "codex"),
        "3": ("GEMINI_API_KEY", "Google Gemini", "api", "gemini-flagship"),
        "4": ("OPENAI_API_KEY", "OpenAI API", "api", "openai-gpt4o"),
        "5": ("ANTHROPIC_API_KEY", "Anthropic API", "api", "anthropic-sonnet"),
        "6": ("OPEN_ROUTER_API_KEY", "OpenRouter", "api", "openrouter-gemini"),
    }
    env_key, provider_name, provider_kind, preset = provider_map[choice]

    api_key = ""
    if provider_kind == "claude-code":
        from artemis.llm.claude import claude_client_status

        ready, detail = claude_client_status()
        if not ready:
            console.print(
                f"\n[bold red]Claude Code is not ready.[/bold red] {detail}\n"
                "For an interactive login, run [bold cyan]claude auth login[/bold cyan]. "
                "For automation, run [bold cyan]claude setup-token[/bold cyan], export "
                "[bold cyan]CLAUDE_CODE_OAUTH_TOKEN[/bold cyan] through your shell or secret "
                "manager, then rerun [bold cyan]artemis init[/bold cyan]."
            )
            raise SystemExit(1)
        console.print("\n[bold green]✔ Claude Code authentication is ready.[/bold green]")
    elif provider_kind == "codex":
        from artemis.llm.codex import codex_client_status

        ready, detail = codex_client_status()
        if not ready:
            console.print(
                f"\n[bold red]Codex is not ready.[/bold red] {detail}\n"
                "Install the Codex CLI if needed, then run [bold cyan]codex login[/bold cyan]. "
                "For a headless terminal, use [bold cyan]codex login --device-auth[/bold cyan]. "
                "Never share the displayed device code. Then rerun "
                "[bold cyan]artemis init[/bold cyan]."
            )
            raise SystemExit(1)
        console.print("\n[bold green]✔ Codex CLI is signed in.[/bold green]")
    else:
        console.print(f"\n[bold]Step 2: Enter your {provider_name} API Key[/bold]")
        api_key = Prompt.ask(f"Enter {env_key}", password=True)
        while not api_key.strip():
            console.print("[bold red]API Key cannot be empty.[/bold red]")
            api_key = Prompt.ask(f"Enter {env_key}", password=True)

    # 3. Scan for Android Devices
    console.print("\n[bold]Step 3: Detecting Android Devices / Emulators...[/bold]")
    detected_devices = []
    adb_path = shutil.which("adb")
    if adb_path:
        try:
            res = subprocess.run(
                [adb_path, "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = [line_str.strip() for line_str in res.stdout.splitlines() if line_str.strip()]
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    serial = parts[0]
                    extra = " ".join(parts[2:])
                    detected_devices.append((serial, extra))
        except (OSError, subprocess.SubprocessError):
            # adb probe failed or timed out; continue setup with no devices.
            pass

    selected_serial = ""
    if detected_devices:
        console.print(
            f"[bold green]✔ Found {len(detected_devices)} connected device(s):[/bold green]"
        )
        for idx, (serial, extra) in enumerate(detected_devices, 1):
            console.print(f"  [{idx}] {serial} ({extra})")
        if len(detected_devices) == 1:
            selected_serial = detected_devices[0][0]
            console.print(f"[dim]Auto-selected device: {selected_serial}[/dim]")
        else:
            dev_choice = Prompt.ask(
                "Select device index",
                choices=[str(i) for i in range(1, len(detected_devices) + 1)],
                default="1",
            )
            selected_serial = detected_devices[int(dev_choice) - 1][0]
    else:
        console.print(
            "[bold yellow]⚠ No connected device detected right now.[/bold yellow]\n"
            "[dim]Artemis will auto-detect your device when you connect it later.[/dim]"
        )

    # 4. Generate .env File
    env_content = [
        "# ==============================================================================",
        "# ☕ ARTEMIS Environment Configuration (Generated by 'artemis init')",
        "# ==============================================================================",
        "",
        "# Android Device Configuration",
        "ADB_HOST=127.0.0.1",
        "ADB_PORT=5037",
    ]
    if env_key is not None:
        env_content.insert(3, f"{env_key}={api_key.strip()}")
    if selected_serial:
        env_content.append(f"ADB_DEVICE_SERIAL={selected_serial}")
    else:
        env_content.append("# ADB_DEVICE_SERIAL=emulator-5554")

    env_content.extend(
        [
            "",
            "# Execution Defaults",
            "ARTEMIS_DEFAULT_PROFILE=pro",
            f"ARTEMIS_LLM_PRESET={preset}",
            "ARTEMIS_TRACES_DIR=./traces",
            "",
        ]
    )

    env_path = get_env_file()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_content), encoding="utf-8")

    # 5. Configure MCP for AI IDEs
    console.print("\n[bold]Step 5: Connect Artemis to your AI IDE (MCP)[/bold]")
    console.print("  [1] Antigravity / Jetski (Recommended - Default)")
    console.print("  [2] Cursor")
    console.print("  [3] Claude Desktop / Claude Code")
    console.print("  [4] Windsurf")
    console.print("  [5] VS Code / Cline / Roo Code")
    console.print("  [6] Codex")
    console.print("  [7] All supported IDEs")
    console.print("  [8] Skip for now")
    mcp_choice = Prompt.ask(
        "Select IDE for MCP auto-install",
        choices=["1", "2", "3", "4", "5", "6", "7", "8"],
        default="1",
    )
    if mcp_choice != "8":
        target_map = {
            "1": "antigravity",
            "2": "cursor",
            "3": "claude",
            "4": "windsurf",
            "5": "vscode",
            "6": "codex",
            "7": "all",
        }
        target = target_map.get(mcp_choice, "antigravity")
        try:
            from artemis.interfaces.cli.commands.mcp import install_mcp_config
            from mcp_server.utils import env_utils

            proj_root = env_utils.get_project_root()
            py_exe = env_utils.resolve_python_executable(proj_root)
            installed_paths = install_mcp_config(target, py_exe, proj_root)
            console.print(
                f"[bold green]✔ Configured Artemis MCP in {len(installed_paths)} IDE location(s).[/bold green]"
            )
        except Exception as e:
            console.print(f"[yellow]Could not auto-install MCP config: {e}[/yellow]")

    console.print()
    console.print(
        Panel(
            f"[bold green]🎉 Setup completed successfully![/bold green]\n\n"
            f"Configured [bold cyan]{provider_name}[/bold cyan] in [dim]{env_path}[/dim]\n\n"
            "Try running your first task:\n"
            '  [bold cyan]uv run artemis run "Open YouTube and search for Lo-Fi Hip Hop"[/bold cyan]\n\n'
            "Or connect/update your AI IDE anytime with:\n"
            "  [bold cyan]uv run artemis mcp --install all[/bold cyan]\n\n"
            "Or run health diagnostics anytime with:\n"
            "  [bold cyan]uv run artemis doctor[/bold cyan]",
            title="☕ Ready to Go!",
            expand=False,
        )
    )
    console.print()
