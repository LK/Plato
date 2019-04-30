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
        this.agentHeading = agentHeading;
        this.agentEnergy = agentEnergy;
        this.agentGunHeat = agentGunHeat;
        this.agentX = agentX;
        this.agentY = agentY;
        this.opponentBearing = opponentBearing;
        this.opponentEnergy = opponentEnergy;
        this.distance = distance;
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

}
