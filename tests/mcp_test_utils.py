from __future__ import annotations

import json
from pathlib import Path
import sys


def write_fake_mcp_server(path: Path) -> Path:
    path.write_text(
        "import json\n"
        "import sys\n"
        "\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        response = {\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': message['id'],\n"
        "            'result': {\n"
        "                'protocolVersion': '2025-11-25',\n"
        "                'capabilities': {'tools': {}},\n"
        "                'serverInfo': {'name': 'fake-mcp', 'version': '0.1.0'},\n"
        "            },\n"
        "        }\n"
        "    elif method == 'notifications/initialized':\n"
        "        continue\n"
        "    elif method == 'tools/list':\n"
        "        response = {\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': message['id'],\n"
        "            'result': {\n"
        "                'tools': [\n"
        "                    {\n"
        "                        'name': 'echo',\n"
        "                        'description': 'Echo input text.',\n"
        "                        'inputSchema': {\n"
        "                            'type': 'object',\n"
        "                            'properties': {'text': {'type': 'string'}},\n"
        "                            'required': ['text'],\n"
        "                        },\n"
        "                    }\n"
        "                ]\n"
        "            },\n"
        "        }\n"
        "    elif method == 'tools/call':\n"
        "        params = message.get('params', {})\n"
        "        name = params.get('name')\n"
        "        arguments = params.get('arguments', {})\n"
        "        if name == 'echo':\n"
        "            response = {\n"
        "                'jsonrpc': '2.0',\n"
        "                'id': message['id'],\n"
        "                'result': {\n"
        "                    'content': [{'type': 'text', 'text': 'echo:' + str(arguments.get('text', ''))}],\n"
        "                    'isError': False,\n"
        "                },\n"
        "            }\n"
        "        else:\n"
        "            response = {\n"
        "                'jsonrpc': '2.0',\n"
        "                'id': message['id'],\n"
        "                'result': {\n"
        "                    'content': [{'type': 'text', 'text': 'unknown tool'}],\n"
        "                    'isError': True,\n"
        "                },\n"
        "            }\n"
        "    else:\n"
        "        response = {\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': message.get('id'),\n"
        "            'error': {'code': -32601, 'message': 'Method not found'},\n"
        "        }\n"
        "    sys.stdout.write(json.dumps(response) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    return path


def write_stateful_fake_mcp_server(path: Path) -> Path:
    path.write_text(
        "import json\n"
        "import sys\n"
        "counter = 0\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'stateful', 'version': '1.0.0'}}\n"
        "    elif method == 'notifications/initialized':\n"
        "        continue\n"
        "    elif method == 'tools/list':\n"
        "        result = {'tools': [{'name': 'increment', 'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}]}\n"
        "    elif method == 'tools/call':\n"
        "        counter += 1\n"
        "        result = {'content': [{'type': 'text', 'text': str(counter)}], 'isError': False}\n"
        "    else:\n"
        "        result = {}\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': message.get('id'), 'result': result}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    return path


def write_oversized_fake_mcp_server(path: Path) -> Path:
    path.write_text(
        "import json\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'oversized', 'version': '1.0.0'}}\n"
        "    elif method == 'notifications/initialized':\n"
        "        continue\n"
        "    else:\n"
        "        result = {'tools': [{'name': 'oversized', 'description': 'x' * 2100000, 'inputSchema': {'type': 'object'}}]}\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': message.get('id'), 'result': result}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    return path


def write_descendant_fake_mcp_server(path: Path, pid_file: Path) -> Path:
    path.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'descendant', 'version': '1.0.0'}}\n"
        "    elif method == 'notifications/initialized':\n"
        "        continue\n"
        "    else:\n"
        "        result = {'tools': []}\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': message.get('id'), 'result': result}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    return path


def write_temp_mcp_app_config(root: Path, server_script: Path) -> Path:
    mcp_config = root / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "fake",
                        "description": "Fake MCP server",
                        "command": sys.executable,
                        "args": [str(server_script)],
                        "enabled": True,
                        "risk_class": "read_only",
                        "capabilities": ["tools"],
                        "allowed_tools": ["echo"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    app_config = root / "app.json"
    app_config.write_text(
        json.dumps(
            {
                "name": "myMoE",
                "mode": "local_model_required",
                "default_moe_config": "tests/fixtures/moe.synthetic.json",
                "language": {
                    "mode": "auto",
                    "respond_in_user_language": True,
                    "supported": ["auto", "en"],
                },
                "runtime": {
                    "auto_configure": True,
                    "preferred_backends": {"fallback": "mlx_lm"},
                    "model_cache_dir": "~/.cache/huggingface",
                    "work_dir": str(root / "runtime"),
                },
                "extensions": {
                    "plugins_dir": "plugins",
                    "skills_dir": "skills",
                    "tools_config": "configs/tools.json",
                    "mcp_config": str(mcp_config),
                    "cron_config": "configs/cron.json",
                },
                "permissions": {
                    "default_write_policy": "approval_required",
                    "allow_process_execution": True,
                    "connector_install_policy": "approval_required",
                    "external_communication_policy": "draft_only",
                },
            }
        ),
        encoding="utf-8",
    )
    return app_config
