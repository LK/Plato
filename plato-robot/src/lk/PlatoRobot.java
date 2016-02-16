package lk;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;

import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class PlatoRobot extends AdvancedRobot {

	final boolean LEARN = true;
	final double EPISILON = LEARN ? 0.3 : 0.0;
	final String UPLOAD_URL = "http://localhost:8080/upload";
	
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
	// It may be a problem that the timing between states isn't consistent, but doing
	// that would mean that we're learning on old data, which is probably even worse.
	public void onScannedRobot(ScannedRobotEvent event) {
		this.write(this.getHeading() + " " + this.getEnergy() + " " + event.getBearing() + " " + event.getEnergy());
	}
	
	public void upload() throws IOException {
		if (!LEARN) return;
		
		String boundary = Long.toHexString(System.currentTimeMillis());
		System.out.println("Opening connection");
		URLConnection connection = new URL(UPLOAD_URL).openConnection();
		System.out.println("Done");
		connection.setDoOutput(true);
		connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
		System.out.println("Set request stuff");
		// http://stackoverflow.com/questions/2469451/upload-files-from-java-client-to-a-http-server
		OutputStream netOutput = connection.getOutputStream();
	    PrintWriter netWriter = new PrintWriter(new OutputStreamWriter(netOutput, "UTF-8"), true);
	    for (File file : this.getDataDirectory().listFiles()) {
	    	System.out.println(file);
	    	netWriter.append("--" + boundary).append("\r\n");
	    	netWriter.append("Content-Disposition: form-data; name=\"textFile\"; filename=\"" + file.getName() + "\"").append("\r\n");
	    	netWriter.append("Content-Type: text/plain; charset=" + "UTF-8").append("\r\n");
	    	netWriter.append("\r\n").flush();
	    	System.out.println(file);
	        Files.copy(file.toPath(), netOutput);
	        System.out.println(file);
	        netOutput.flush();
	        netWriter.append("\r\n").flush();
	        System.out.println(file);
	    }
	    netWriter.append("--" + boundary + "--").append("\r\n").flush();
	    System.out.println("Sent files");
	    int responseCode = ((HttpURLConnection)connection).getResponseCode();
	    System.out.println(responseCode);
	    if (responseCode != 200) {
	    	System.out.println("RECEIVED " + responseCode + " WHEN UPLOADING FILES");
	    } else {
	    	for (File file : this.getDataDirectory().listFiles()) {
	    		file.delete();
	    	}
	    }
	}
	
	public void onWin(WinEvent event) {
		this.write("W");
		try {
			this.output.close();
			this.upload();
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
