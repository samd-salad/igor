# Home Assistant MCP Server — One-Time Setup

Igor reaches HA via HA's built-in MCP Server integration. Set this up ONCE
on the HA box, then never touch it.

## 1. Install the integration

HA UI → Settings → Devices & Services → Add Integration → search "Model
Context Protocol Server" → install.

When prompted for auth, choose **Long-lived access token** and paste the
existing `HA_TOKEN` value (the one already in Portainer's Igor stack env).

The endpoint Igor connects to is automatically:
`http://10.0.40.5:8123/api/mcp`

## 2. Expose the entities Igor should control

Igor sees only entities that HA flags as exposed-to-voice. This is HA's
expose toggle, NOT Igor's allowlist — it's the safety surface.

HA UI → Settings → Voice assistants → Expose. Toggle on every light,
switch, media_player, climate, todo, etc. that Igor should be able to act on.

## 3. Verify Igor can see them

After redeploying Igor's container, on the Pi:

```bash
docker logs igor | grep "HA MCP catalog"
```

Expected: a line like `HA MCP catalog: 17 tool(s) cached`. If it shows `0`,
HA's MCP integration isn't reachable — check the endpoint URL and the
exposure list.

## 4. Smoke tests

Say:
- "Okay Nabu, turn off the kitchen lights"
- "Okay Nabu, what's the weather?"  (Igor-native, doesn't need HA)
- "Okay Nabu, remember I prefer dark roast coffee"  (Igor-native)

Then on the Pi:

```bash
docker exec igor python -m server.tools.recent_episodes 5
```

Should show all three with `tools: HassTurnOff` (or similar) / `get_weather`
/ `save_memory` in the tool-calls line.
