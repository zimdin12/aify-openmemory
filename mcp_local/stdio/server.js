/**
 * Host-side MCP Server (stdio transport)
 *
 * Mirrors the SSE server's tools, calling the containerized service via HTTP.
 * Use when the MCP client doesn't support SSE transport.
 *
 * Install:
 *   cd mcp/stdio && npm install
 *   claude mcp add my-service -- node /path/to/mcp/stdio/server.js
 *
 * Env vars:
 *   SERVICE_API_URL  - Base URL (default: http://localhost:8800)
 *   SERVICE_USER_ID  - User identity (default: default)
 *   SERVICE_APP_NAME - App name (default: claude-code)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const API_URL = process.env.SERVICE_API_URL || "http://localhost:8800";

const server = new McpServer({
  name: "agentify-service",
  version: "0.1.0",
});

async function apiCall(method, path, body = null) {
  const url = `${API_URL}${path}`;
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) options.body = JSON.stringify(body);
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text}`);
  }
  return response.json();
}

function ok(data) {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

function err(error) {
  return { content: [{ type: "text", text: `Error: ${error.message}` }], isError: true };
}

// --- Service tools ---

server.tool(
  "service_info",
  "Get service info, available containers, endpoints, and capabilities",
  {},
  async () => {
    try { return ok(await apiCall("GET", "/info")); }
    catch (e) { return err(e); }
  }
);

server.tool(
  "service_health",
  "Check if the service is healthy and all dependencies are available",
  {},
  async () => {
    try { return ok(await apiCall("GET", "/health")); }
    catch (e) { return err(e); }
  }
);

// --- Container management tools ---

server.tool(
  "list_containers",
  "List all managed sub-containers, their status, GPU allocation, and URLs",
  {},
  async () => {
    try { return ok(await apiCall("GET", "/api/v1/containers")); }
    catch (e) { return err(e); }
  }
);

server.tool(
  "start_container",
  "Start a managed sub-container by name. Auto-resolves shared containers.",
  { name: z.string().describe("Container name to start") },
  async ({ name }) => {
    try { return ok(await apiCall("POST", `/api/v1/containers/${name}/start`)); }
    catch (e) { return err(e); }
  }
);

server.tool(
  "stop_container",
  "Stop a running sub-container by name",
  { name: z.string().describe("Container name to stop") },
  async ({ name }) => {
    try { return ok(await apiCall("POST", `/api/v1/containers/${name}/stop`)); }
    catch (e) { return err(e); }
  }
);

server.tool(
  "gpu_status",
  "Get GPU device allocation showing which containers use which GPUs and memory fractions",
  {},
  async () => {
    try { return ok(await apiCall("GET", "/api/v1/gpu")); }
    catch (e) { return err(e); }
  }
);

server.tool(
  "container_logs",
  "Get recent logs from a managed sub-container",
  {
    name: z.string().describe("Container name"),
    tail: z.number().optional().default(50).describe("Number of log lines"),
  },
  async ({ name, tail }) => {
    try { return ok(await apiCall("GET", `/api/v1/containers/${name}/logs?tail=${tail}`)); }
    catch (e) { return err(e); }
  }
);

// --- TODO: Add service-specific tools ---

const transport = new StdioServerTransport();
await server.connect(transport);
