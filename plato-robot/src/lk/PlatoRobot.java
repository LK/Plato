package lk;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class PlatoRobot extends AdvancedRobot {

	StateReporter stateReporter;
	Network network;

	boolean surrender = false;
	File networkFile;

	State lastState;
	double lastBearing;
	double lastEnergy;
	Action lastAction;
	double lastReward;

	private enum Action {
		FORWARD, BACKWARD, LEFT, RIGHT, FIRE, NOTHING;

		public static Action fromInteger(int x) {
			switch (x) {
			case 0:
				return FORWARD;
			case 1:
				return BACKWARD;
			case 2:
				return LEFT;
			case 3:
				return RIGHT;
			case 4:
				return FIRE;
			case 5:
				return NOTHING;
			}
			return null;
		}
	}

	public void run() {
		this.surrender = false;
		this.stateReporter = new StateReporter("localhost", 8000);
		this.network = new Network();
		long start = System.currentTimeMillis();
		Random rand = new Random();
		this.networkFile = this.getDataFile("network" + rand.nextInt() + ".hdf5");
		// this.networkFile = this.getDataFile("network.hfd5");
		this.network.downloadNetwork("http://localhost:8001", this.networkFile);
		long end = System.currentTimeMillis();

		System.out.println("Loading network took " + (end - start) + " ms");
		System.out.println("Time is " + this.getTime());

		while (true) {
			this.setTurnRadarRight(360);
			this.execute();
			// if (this.getTime() >= 2000 && !this.surrender) {
			// this.surrender = true;
			// State newState = new State((float) this.getHeading(), (float)
			// this.getEnergy(), (float) lastBearing,
			// (float) lastEnergy);
			// this.stateReporter.recordTransition(lastAction.ordinal(), -1.0f, newState,
			// true);
			// this.stateReporter.close();
			// } else if (!this.surrender && this.getTime() % 10 == 0) {
			if (this.getTime() % 10 == 0 && this.getEnergy() > 0) {
				this.performAction();
			}

			if (this.getTime() % 1000 == 0) {
				System.out.println("Loading network at " + this.getTime());
				this.networkFile.delete();
				this.network.downloadNetwork("http://localhost:8001", this.networkFile);
				System.out.println("Loaded network at " + this.getTime());
			}
		}
	}

	public void performAction() {
		if (this.lastState == null) {
			return;
		}

		double[] inputs = { this.lastState.agentHeading, this.lastState.agentEnergy, this.lastState.agentGunHeat,
				this.lastState.agentX, this.lastState.agentY, this.lastState.opponentBearing,
				this.lastState.opponentEnergy, this.lastState.distance };
		double[] policy = this.network.evaluate(inputs);

		double eps = -0.9 / 250000 * this.network.updates + 1;
		eps = Math.max(eps, 0.1);

		if (this.getTime() <= 10) {
			System.out.println("eps: " + eps);
		}

		Random rand = new Random();
		Action action = Action.NOTHING;
		if (rand.nextDouble() < eps) {
			// Take random action.
			int r = rand.nextInt(Action.values().length);
			action = Action.fromInteger(r);

		} else {
			// Take greedy action.
			double maxQ = -999999.0;
			for (int i = 0; i < policy.length; i++) {
				if (policy[i] > maxQ) {
					maxQ = policy[i];
					action = Action.fromInteger(i);
				}
			}
		}

		switch (action) {
		case FORWARD:
			this.setAhead(10);
			break;
		case BACKWARD:
			this.setBack(10);
			break;
		case LEFT:
			this.setTurnLeft(10);
			break;
		case RIGHT:
			this.setTurnRight(10);
			break;
		case FIRE:
			this.setFire(1);
			break;
		default:
			break;
		}

		this.lastAction = action;

		if (!this.surrender) {
			this.stateReporter.recordTransition(action.ordinal(), (float) this.lastReward, this.lastState, false);
		}
	}

	public void onScannedRobot(ScannedRobotEvent event) {
		lastReward = this.lastState != null
				? 2 * (this.lastState.opponentEnergy - event.getEnergy()) + this.getEnergy()
						- this.lastState.agentEnergy
				: 0.0f;
		System.out.println("Reward: " + lastReward);
		lastState = new State((float) this.getHeading(), (float) this.getEnergy(), (float) this.getGunHeat(),
				(float) this.getX(), (float) this.getY(), (float) event.getBearing(), (float) event.getEnergy(),
				(float) event.getDistance());

		// this.lastBearing = event.getBearing();
		// this.lastEnergy = event.getEnergy();
		// this.lastAction = action;
		// // TODO: reward = 2 * their energy loss - our energy loss
		// // TODO: report on regular interval
		// if (!this.surrender) {
		// State newState = new State((float) this.getHeading(), (float)
		// this.getEnergy(), (float) event.getBearing(),
		// (float) event.getEnergy());
		// this.stateReporter.recordTransition(action.ordinal(), 0.0f, newState, false);
		// }
	}

	public void onDeath(DeathEvent event) {
		if (!this.surrender) {
			this.stateReporter.recordTransition(this.lastAction.ordinal(), 0.0f, this.lastState, true);
			this.stateReporter.close();
		}
		this.networkFile.delete();
	}

	public void onWin(WinEvent event) {
		if (!this.surrender) {
			this.stateReporter.recordTransition(this.lastAction.ordinal(), 0.0f, this.lastState, true);
			this.stateReporter.close();
		}
		this.networkFile.delete();
	}

}
