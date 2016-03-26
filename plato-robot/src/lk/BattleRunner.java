package lk;

import java.io.File;

import robocode.control.BattleSpecification;
import robocode.control.BattlefieldSpecification;
import robocode.control.RobocodeEngine;
import robocode.control.RobotSpecification;
import robocode.control.events.BattleAdaptor;

public class BattleRunner extends BattleAdaptor {

	public static void main(String[] args) {
		BattleRunner runner = new BattleRunner();
		runner.run(1000);
		
	}
	
	public BattleRunner() {
		
	}
	
	public void run(int rounds) {
        RobocodeEngine engine = new RobocodeEngine(new File("/Users/Lenny/robocode"));
        engine.addBattleListener(this);

        BattlefieldSpecification battlefield = new BattlefieldSpecification(800, 600); // 800x600
        RobotSpecification[] selectedRobots = engine.getLocalRepository("lk.PlatoRobot,sample.SittingDuck");
        BattleSpecification battleSpec = new BattleSpecification(rounds, battlefield, selectedRobots);
        
        engine.setVisible(true);
        
        engine.runBattle(battleSpec, true);
        engine.close();

        System.exit(0);
	}
	
}
