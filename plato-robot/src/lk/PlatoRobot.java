package lk;

import robocode.AdvancedRobot;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;

public class PlatoRobot extends AdvancedRobot {
	
	StateReporter stateReporter; 
	
	public void run() {
		this.stateReporter = new StateReporter("localhost", 8000);
		
		while (true) {
			this.setTurnRadarRight(360);
			this.execute();

			System.out.println(this.getTime());
		}
	}
	
	public void onScannedRobot(ScannedRobotEvent event) {
		this.stateReporter.report((float)this.getHeading(), (float)this.getEnergy(), (float)event.getBearing(), (float)event.getEnergy(), 0.0f);
	}
	
	public void onRoundEnded(RoundEndedEvent event) {
		this.stateReporter.close();
	}
	
}
