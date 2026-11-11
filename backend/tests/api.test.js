const { test, describe, before, after } = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const app = require("../src/server");

let server;
let baseUrl;

function request(method, path, body = null, token = null) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, baseUrl);
    const headers = {};
    let payload = null;

    if (body) {
      payload = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(payload);
    }
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const req = http.request(
      url,
      { method, headers },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve({ status: res.statusCode, body: JSON.parse(data || "{}") });
          } catch {
            resolve({ status: res.statusCode, body: data });
          }
        });
      }
    );

    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

describe("Express Backend API Gateway (Phase 3)", () => {
  before(
    () =>
      new Promise((resolve) => {
        server = app.listen(0, "127.0.0.1", () => {
          const { port } = server.address();
          baseUrl = `http://127.0.0.1:${port}`;
          resolve();
        });
      })
  );

  after(
    () =>
      new Promise((resolve) => {
        if (server) server.close(resolve);
        else resolve();
      })
  );

  test("GET /api/health returns service status", async () => {
    const res = await request("GET", "/api/health");
    assert.equal(res.status, 200);
    assert.equal(res.body.status, "ok");
    assert.equal(res.body.service, "express-api-gateway");
  });

  test("Authentication lifecycle: register, login, and protected /me route", async () => {
    const uniqueEmail = `test_${Date.now()}@example.com`;

    // 1. Register
    const reg = await request("POST", "/api/auth/register", {
      name: "Test User",
      email: uniqueEmail,
      password: "securepassword123",
    });
    assert.equal(reg.status, 201);
    assert.ok(reg.body.token);
    assert.equal(reg.body.user.email, uniqueEmail);

    // 2. Protected route rejected without token
    const unauth = await request("GET", "/api/auth/me");
    assert.equal(unauth.status, 401);

    // 3. Login
    const login = await request("POST", "/api/auth/login", {
      email: uniqueEmail,
      password: "securepassword123",
    });
    assert.equal(login.status, 200);
    assert.ok(login.body.token);

    // 4. Protected /me works with valid token
    const me = await request("GET", "/api/auth/me", null, login.body.token);
    assert.equal(me.status, 200);
    assert.equal(me.body.user.email, uniqueEmail);

    // 5. Protected /api/photos works with valid token
    const photos = await request("GET", "/api/photos", null, login.body.token);
    assert.equal(photos.status, 200);
    assert.ok(Array.isArray(photos.body.photos));
  });

  test("GET /api/ai/styles returns available photographic presets", async () => {
    const res = await request("GET", "/api/ai/styles");
    assert.equal(res.status, 200);
    assert.ok(Array.isArray(res.body.styles));
    assert.ok(res.body.styles.length >= 7);
  });
});

