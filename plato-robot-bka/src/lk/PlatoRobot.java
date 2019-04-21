package lk;

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
	
	double lastBearing;
	double lastEnergy;
	Action lastAction;
	
	private enum Action {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		FIRE,
		NOTHING;
		
		public static Action fromInteger(int x) {
	        switch(x) {
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
		System.out.println("#####");

		this.surrender = false;
		this.stateReporter = new StateReporter("localhost", 8000);
		this.network = new Network();
		this.network.downloadNetwork("http://localhost:8001", this.getDataFile("network.hdf5"));
		
		while (true) {
			this.setTurnRadarRight(360);
			this.execute();
			
			if (this.getTime() >= 2000 && !this.surrender) {
				this.surrender = true;
				this.stateReporter.report((float)this.getHeading(), (float)this.getEnergy(), (float)lastBearing, (float)lastEnergy, -1.0f, lastAction.ordinal());
				this.stateReporter.close();
			}
		}
	}
	
	public void onScannedRobot(ScannedRobotEvent event) {
		double[] inputs = {this.getHeading(), this.getEnergy(), event.getBearing(), event.getEnergy()};
		double[] policy = this.network.policy(inputs);
		
		System.out.println(this.getTime());
		
		Random rand = new Random();
		double r = rand.nextDouble();
		double sum = 0.0;
		Action action = Action.NOTHING;
		for (int i = 0; i < policy.length; i++) {
			sum += policy[i];
			if (r < sum) {
				action = Action.fromInteger(i);
				break;
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
		
		this.lastBearing = event.getBearing();
		this.lastEnergy = event.getEnergy();
		this.lastAction = action;
		if (!this.surrender) this.stateReporter.report((float)this.getHeading(), (float)this.getEnergy(), (float)event.getBearing(), (float)event.getEnergy(), 0.0f, action.ordinal());
	}
	
	public void onDeath(DeathEvent event) {
		if (!this.surrender) this.stateReporter.report((float)this.getHeading(), (float)this.getEnergy(), (float)lastBearing, (float)lastEnergy, -1.0f, lastAction.ordinal());
		if (!this.surrender) this.stateReporter.close();
	}
	
	public void onWin(WinEvent event) {
		if (!this.surrender) this.stateReporter.report((float)this.getHeading(), (float)this.getEnergy(), (float)lastBearing, (float)lastEnergy, 1.0f, lastAction.ordinal());
		if (!this.surrender) this.stateReporter.close();
	}
	
}
