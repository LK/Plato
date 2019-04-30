package lk;

import java.nio.ByteBuffer;

public class State {
    float agentHeading;
    float agentEnergy;
    float agentGunHeat;
    float agentX;
    float agentY;
    float opponentBearing;
    float opponentEnergy;
    float distance;

    public State(float agentHeading, float agentEnergy, float agentGunHeat, float agentX, float agentY,
            float opponentBearing, float opponentEnergy, float distance) {
        this.agentHeading = agentHeading / 18.0f;
        this.agentEnergy = agentEnergy / 10.0f;
        this.agentGunHeat = agentGunHeat / .3f;
        this.agentX = agentX / 80.0f;
        this.agentY = agentY / 60.0f;
        this.opponentBearing = opponentBearing / 18.0f;
        this.opponentEnergy = opponentEnergy / 10.0f;
        this.distance = distance / 20.0f;
    }

    public static int size() {
        return 4 * 8;
    }

    public void writeToBuffer(ByteBuffer buf) {
        buf.putFloat(agentHeading);
        buf.putFloat(agentEnergy);
        buf.putFloat(agentGunHeat);
        buf.putFloat(agentX);
        buf.putFloat(agentY);
        buf.putFloat(opponentBearing);
        buf.putFloat(opponentEnergy);
        buf.putFloat(distance);
    }

    public String toString() {
        String res = "-----";
        res += "\nagentHeading: " + this.agentHeading;
        res += "\nagentEnergy: " + this.agentEnergy;
        res += "\nagentGunHeat: " + this.agentGunHeat;
        res += "\nagentX: " + this.agentX;
        res += "\nagentY: " + this.agentY;
        res += "\nopponentBearing: " + this.opponentBearing;
        res += "\nopponentEnergy: " + this.opponentEnergy;
        res += "\ndistance: " + this.distance;
        return res + "\n-----";
    }

}
