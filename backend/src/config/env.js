const path = require("path");
require("dotenv").config();

const ROOT_DIR = path.resolve(__dirname, "../../../");

module.exports = {
  PORT: process.env.PORT || 5000,
  JWT_SECRET: process.env.JWT_SECRET || "ai_photo_gen_phase3_secret_key",
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || "7d",
  PYTHON_API_URL: process.env.PYTHON_API_URL || "http://127.0.0.1:8000",
  ROOT_DIR,
  RAW_PHOTOS_DIR: path.join(ROOT_DIR, "data", "raw"),
  GENERATED_DIR: path.join(ROOT_DIR, "outputs", "generated"),
  DB_PATH: path.resolve(__dirname, "../../data/db.json"),
};
