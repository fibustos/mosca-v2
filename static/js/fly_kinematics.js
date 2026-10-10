import * as THREE from "three";

const MAX_REACH_ANGLE = Math.PI / 4;

function unitInterval(value) {
  return Number.isFinite(value) ? THREE.MathUtils.clamp(value, 0, 1) : 0;
}

export class FlyKinematics {
  constructor(legJoints) {
    const ganglionByLeg = {
      FL: "T1_L", FR: "T1_R",
      ML: "T2_L", MR: "T2_R",
      BL: "T3_L", BR: "T3_R",
    };
    this.legs = Object.entries(ganglionByLeg).map(([leg, ganglion]) => {
      const { femur, tibia } = legJoints[leg];
      return {
        femur,
        tibia,
        ganglion,
        extension: 0,
        femurProceduralAngle: femur.rotation.x,
        tibiaProceduralAngle: tibia.rotation.x,
      };
    });
    this.vncLegsNeuro = {};
  }

  updateFromTelemetry(telemetry) {
    const activity = telemetry?.vnc_legs_neuro;
    if (activity && typeof activity === "object") {
      this.vncLegsNeuro = activity;
      return;
    }
    const legacy = telemetry?.front_legs_neuro;
    this.vncLegsNeuro = legacy && typeof legacy === "object"
      ? {
        T1_L: { ext: legacy.FL_ext, flex: legacy.FL_flex },
        T1_R: { ext: legacy.FR_ext, flex: legacy.FR_flex },
      }
      : {};
  }

  resetRestPose() {
    this.vncLegsNeuro = {};
    for (const leg of this.legs) {
      leg.extension = 0;
      leg.femur.rotation.x = leg.femurProceduralAngle;
      leg.tibia.rotation.x = leg.tibiaProceduralAngle;
    }
  }

  update(elapsed) {
    const smoothing = 1 - Math.exp(-elapsed * 12);
    for (const leg of this.legs) {
      const ganglion = this.vncLegsNeuro[leg.ganglion] ?? {};
      const extensorDrive = unitInterval(ganglion.ext);
      const flexorDrive = unitInterval(ganglion.flex);
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
