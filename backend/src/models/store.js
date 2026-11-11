const fs = require("fs");
const path = require("path");
const { DB_PATH } = require("../config/env");

class JsonStore {
  constructor(dbPath = DB_PATH) {
    this.dbPath = dbPath;
    this._ensureDb();
  }

  _ensureDb() {
    const dir = path.dirname(this.dbPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    if (!fs.existsSync(this.dbPath)) {
      this._write({ users: [], gallery: [] });
    }
  }

  _read() {
    try {
      const raw = fs.readFileSync(this.dbPath, "utf-8");
      return JSON.parse(raw);
    } catch {
      return { users: [], gallery: [] };
    }
  }

  _write(data) {
    fs.writeFileSync(this.dbPath, JSON.stringify(data, null, 2), "utf-8");
  }

  findUserByEmail(email) {
    const db = this._read();
    return db.users.find((u) => u.email.toLowerCase() === email.toLowerCase()) || null;
  }

  findUserById(id) {
    const db = this._read();
    return db.users.find((u) => u.id === id) || null;
  }

  createUser({ name, email, passwordHash }) {
    const db = this._read();
    const user = {
      id: `usr_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      name,
      email: email.toLowerCase(),
      passwordHash,
      createdAt: new Date().toISOString(),
    };
    db.users.push(user);
    this._write(db);
    return user;
  }

  addGalleryItem(item) {
    const db = this._read();
    const record = {
      id: `img_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      createdAt: new Date().toISOString(),
      ...item,
    };
    db.gallery.unshift(record);
    this._write(db);
    return record;
  }

  getGalleryItems() {
    const db = this._read();
    return db.gallery || [];
  }
}

module.exports = new JsonStore();
module.exports.JsonStore = JsonStore;
