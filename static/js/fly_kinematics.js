import * as THREE from "three";

const MAX_REACH_ANGLE = Math.PI / 4;

function unitInterval(value) {
  return Number.isFinite(value) ? THREE.MathUtils.clamp(value, 0, 1) : 0;
}

export class FlyKinematics {
  constructor({ frontLeftFemur, frontLeftTibia, frontRightFemur, frontRightTibia }) {
    this.frontLegs = [
      {
        femur: frontLeftFemur,
        tibia: frontLeftTibia,
        extension: 0,
        femurProceduralAngle: frontLeftFemur.rotation.x,
        tibiaProceduralAngle: frontLeftTibia.rotation.x,
        extKey: "FL_ext",
        flexKey: "FL_flex",
      },
      {
        femur: frontRightFemur,
        tibia: frontRightTibia,
        extension: 0,
        femurProceduralAngle: frontRightFemur.rotation.x,
        tibiaProceduralAngle: frontRightTibia.rotation.x,
        extKey: "FR_ext",
        flexKey: "FR_flex",
      },
    ];
    this.frontLegsNeuro = {};
  }

  updateFromTelemetry(telemetry) {
    const activity = telemetry?.front_legs_neuro;
    this.frontLegsNeuro = activity && typeof activity === "object" ? activity : {};
  }

  resetRestPose() {
    this.frontLegsNeuro = {};
    for (const leg of this.frontLegs) {
      leg.extension = 0;
      leg.femur.rotation.x = leg.femurProceduralAngle;
      leg.tibia.rotation.x = leg.tibiaProceduralAngle;
    }
  }

  update(elapsed) {
    const smoothing = 1 - Math.exp(-elapsed * 12);
    for (const leg of this.frontLegs) {
      const extensorDrive = unitInterval(this.frontLegsNeuro[leg.extKey]);
      const flexorDrive = unitInterval(this.frontLegsNeuro[leg.flexKey]);
      const extension = (extensorDrive - flexorDrive * .72) * MAX_REACH_ANGLE;
      leg.extension = THREE.MathUtils.lerp(leg.extension, extension, smoothing);
      leg.femur.rotation.x = THREE.MathUtils.lerp(
        leg.femur.rotation.x,
        leg.femurProceduralAngle + leg.extension,
        smoothing,
      );
      leg.tibia.rotation.x = THREE.MathUtils.lerp(
        leg.tibia.rotation.x,
        leg.tibiaProceduralAngle - leg.extension * .5,
        smoothing,
      );
    }
  }
}
