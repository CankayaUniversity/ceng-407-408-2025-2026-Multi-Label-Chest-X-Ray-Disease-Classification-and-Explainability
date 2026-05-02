const mongoose = require("mongoose");

const DoctorSchema = new mongoose.Schema({
  fullName: { type: String, required: true },
  email: { type: String, unique: true, required: true },
  password: { type: String, required: true },
  hospital: { type: mongoose.Schema.Types.ObjectId, ref: "hospitals" },
  status: {
    type: String,
    enum: ["pending", "approved", "rejected"],
    default: "pending",
  },
});

module.exports = mongoose.model("doctors", DoctorSchema);
