package lk;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;

import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class PlatoRobot extends AdvancedRobot {

	final boolean LEARN = true;
	
	private BufferedWriter output;
	
	public void run() {
		try {
			File file = this.getDataFile(this.getCurrentTimeStamp() + ".txt");
			file.createNewFile();
			FileOutputStream outputStream = new FileOutputStream(file);
			this.output = new BufferedWriter(new OutputStreamWriter(outputStream));
		} catch (Exception e) {
			System.out.println("COULD NOT CREATE FILE");
		}
		
		while (true) {
			this.setTurnRadarRight(360);
			this.execute();
		}
	}
	
	// When we scan a robot, record a new state.
	public void onScannedRobot(ScannedRobotEvent event) {
		this.write(this.getHeading() + " " + this.getEnergy() + " " + event.getBearing() + " " + event.getEnergy());
	}
	
	public void onWin(WinEvent event) {
		this.write("W");
		try {
			this.output.close();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE/UPLOAD FILE");
		}
	}
	
	public void onDeath(DeathEvent event) {
		this.write("L");
		try {
			this.output.close();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE TO FILE");
		}
	}
	
	public String getCurrentTimeStamp() {
	    return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
	}
	
	public void write(String s) {
		try {
			this.output.write(s);
			this.output.newLine();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE TO FILE");
		}
	}
}
