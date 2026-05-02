const mongoose = require("mongoose");

const HospitalSchema = new mongoose.Schema({
  name: { type: String, required: true },
  city: { type: String, required: true },
});

module.exports = mongoose.model("hospitals", HospitalSchema);
