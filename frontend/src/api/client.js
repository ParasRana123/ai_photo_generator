const TOKEN_KEY = "persona_ai_token";

export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

async function apiRequest(endpoint, options = {}) {
  const token = getStoredToken();
  const headers = { ...(options.headers || {}) };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Only set JSON content-type when body is not FormData
  if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(endpoint, {
    ...options,
    headers,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.detail || `Request failed with status ${response.status}`);
  }
  return data;
}

export const authApi = {
  register: (payload) =>
    apiRequest("/api/auth/register", { method: "POST", body: JSON.stringify(payload) }),
  login: (payload) =>
    apiRequest("/api/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  me: () => apiRequest("/api/auth/me"),
};

export const photosApi = {
  list: () => apiRequest("/api/photos"),
  upload: (files) => {
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("photos", file));
    return apiRequest("/api/photos/upload", { method: "POST", body: formData });
  },
  remove: (filename) =>
    apiRequest(`/api/photos/${encodeURIComponent(filename)}`, { method: "DELETE" }),
};

export const aiApi = {
  getStyles: () => apiRequest("/api/ai/styles"),
  startTraining: (payload) =>
    apiRequest("/api/ai/train", { method: "POST", body: JSON.stringify(payload) }),
  getTrainStatus: () => apiRequest("/api/ai/train/status"),
  generate: (payload) =>
    apiRequest("/api/ai/generate", { method: "POST", body: JSON.stringify(payload) }),
  getGallery: () => apiRequest("/api/ai/gallery"),
};
