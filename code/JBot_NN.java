package JBot_NN;

import java.awt.*;
import java.io.IOException;
import java.io.PrintStream;

import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class JBot_NN extends AdvancedRobot {
	int action_old, action;
	double[] states_old = new double[5];
	double[] states = new double[5];
	double reward;
	double oppoDist, oppoBearing;
	boolean found = false;
	double epsilon = 0;  // when 1.0 always random
	boolean useInterReward = true;  // when false only collects terminal rewards
	boolean train = true;  // when false JBot does not learn, i.e. LUT does not update
//	double interRewardGood = 1.4;
//	double interRewardBad = 0.8;
//	double terminalRewardWin = 25;
//	double terminalRewardDeath = 15;

	// corner
//	NeuralNet myNet = new NeuralNet(10, 20, 0.2, 0.9, -1, 1);
//	NeuralNet myNet = new NeuralNet(10, 15, 0.045, 0.35, -1, 1);
//	NeuralNet myNet = new NeuralNet(10, 100, 0.005, 0.6, -1, 1);
//	NeuralNet myNet = new NeuralNet(10, 100, 0.2, 0.1, -1, 1);
//	NeuralNet myNet = new NeuralNet(10, 100, 0.002, 0.9, -1, 1);
//	NeuralNet myNet = new NeuralNet(7, 100, 0.02, 0.6, -1, 1);
//	NeuralNet myNet = new NeuralNet(7, 100, 0.005, 0.5, -1, 1);  // corner
//	NeuralNet myNet = new NeuralNet(7, 40, 0.005, 0.05, -1, 1);  // corner linear

	// fire
//	NeuralNet myNet = new NeuralNet(7, 100, 0.05, 0.1, -1, 1);
	NeuralNet myNet = new NeuralNet(7, 40, 0.05, 0.1, -1, 1);  // fire linear
	public void run() {  // off policy
		if (!train) epsilon = 0;
		loadData();
		setColors(Color.darkGray, Color.black, Color.lightGray);
	    setAdjustGunForRobotTurn(false);  // when false gun is fixed to body
	    setAdjustRadarForGunTurn(false);  // when false radar is fixed to body
	    turnRadarRightRadians(2 * Math.PI);

    	updateState();
	    while (true) {
	    	updateOldState();
	    	action = myNet.pickAction(states, Math.random() < epsilon);
			reward = 0.0;
	    	switch (action)
  			{
  				case Action.frontLeft:
  					setAhead(Action.forewardDist);
  					setTurnLeft(Action.turnDegree);
  					break;
  				case Action.frontRight:
  					setAhead(Action.forewardDist);
  					setTurnRight(Action.turnDegree);
  					break;
  				case Action.backLeft:
  					setBack(Action.backwardDist);
  					setTurnRight(Action.turnDegree);
  					break;
  				case Action.backRight:
  					setBack(Action.backwardDist);
  					setTurnLeft(Action.turnDegree);
  					break;
  				case Action.fire:
  					ahead(0);
  					turnLeft(0);
  					scanAndFire();
  					break;
  			}
	    	execute();
	    	while (getDistanceRemaining() != 0 || getTurnRemaining() != 0) execute();
		    turnRadarRightRadians(2 * Math.PI);
	    	updateState();
	    	if (train) myNet.updateNNOff(states_old, states, action, reward);
	    }
	}
	
//	public void run() {  // on policy
//		if (!train) epsilon = 0;
//		loadData();
//		setColors(Color.darkGray, Color.black, Color.lightGray);
//	    setAdjustGunForRobotTurn(false);  // when false gun is fixed to body
//	    setAdjustRadarForGunTurn(false);  // when false radar is fixed to body
//	    
//	    turnRadarRightRadians(2 * Math.PI);
//    	updateState();
//    	action = lut.pickAction(state, Math.random() < epsilon);
//    	state_old = state;
//    	action_old = action;
//	    while (true) {
//	    	reward = 0.0;
//	    	switch (action)
//  			{ 
//  				case Action.frontLeft:
//  					setAhead(Action.forewardDist);
//  					setTurnLeft(Action.turnDegree);
//  					break;
//  				case Action.frontRight:
//  					setAhead(Action.forewardDist);
//  					setTurnRight(Action.turnDegree);
//  					break;
//  				case Action.backLeft:
//  					setBack(Action.backwardDist);
//  					setTurnRight(Action.turnDegree);
//  					break;
//  				case Action.backRight:
//  					setBack(Action.backwardDist);
//  					setTurnLeft(Action.turnDegree);
//  					break;
//  				case Action.fire:
//  					ahead(0);
//  					turnLeft(0);
//  					scanAndFire();
//  					break;
//  			}
//	    	execute();
//	    	while (getDistanceRemaining() != 0 || getTurnRemaining() != 0) execute();
//	    	turnRadarRightRadians(2 * Math.PI);
//	    	updateState();
//	    	action = lut.pickAction(state, Math.random() < epsilon);
//	    	if (train) lut.updateLUTOn(state_old, state, action_old, action, reward);
//	    	state_old = state;
//	    	action_old = action;
//	    }
//	}
	
	public void scanAndFire() {
		found = false;
		while (!found) {
			setTurnRadarLeft(360);
			execute();
		}
		turnGunLeft(getGunHeading() - getHeading() - oppoBearing);
		double currentOppoDist = oppoDist;
		if (currentOppoDist < 101) fire(6);
		else if (currentOppoDist < 201) fire(4);
		else if (currentOppoDist < 301) fire(2);
		else fire(1);
	}
	
	private void updateState() {
		this.states[0] = States.getHeading(getHeadingRadians());  // heading
		this.states[1] = States.getTargetDistance(oppoDist);  // targetDistance
		this.states[2] = States.getTargetBearing(oppoBearing);  // targetBearing
		this.states[3] = States.getXPosition(getX());  // xPosition
		this.states[4] = States.getYPosition(getY());  // yPosition
	}
	
	private void updateOldState() {
		for (int i = 0; i < this.states.length; i++) this.states_old[i] = this.states[i];
	}
	
	//------------------------Events---------------------------------
	public void onScannedRobot(ScannedRobotEvent e) {
		oppoDist = e.getDistance();
		oppoBearing = e.getBearing();
		found = true;
	}
	
	public void onBulletHit(BulletHitEvent e) {
		if (useInterReward) reward += 7.5 * e.getBullet().getPower();
	}
	
	public void onBulletMissed(BulletMissedEvent e) {
		if (useInterReward) reward -= 8 * e.getBullet().getPower();
	}
	
	public void onHitByBullet(HitByBulletEvent e) {
		if (useInterReward) reward -= e.getBullet().getPower() * 8.5;
	}
	
	public void onHitWall(HitWallEvent e) {
		if (useInterReward) reward -= 10;
	}
	
	public void onWin(WinEvent event) {
		if (train) saveData();
		reward += 100;
//		System.out.print("wwwwwwwwwwww");
		if (train) saveBattleHist(1);
	}
	
	public void onDeath(DeathEvent event) {
		if (train) saveData();
		reward -= 75;
//		System.out.print("ddddddddddddd");
		if (train) saveBattleHist(0);
	}
	
	//------------------------File IO---------------------------------
	public void loadData() { 
		try {
			myNet.loadFile(getDataFile("NN.dat"));
	    }
		catch (Exception e) {
		}
	}
	
	public void saveData() { 
		try {
			myNet.save(getDataFile("NN.dat"));
		}
		catch (Exception e) {
			out.println("Exception trying to write: " + e);
		}
	}
	
	public void saveBattleHist(int win) {
		PrintStream write = null;
		try {
			write = new PrintStream(new RobocodeFileOutputStream(getDataFile("battle_history_NN.dat").getAbsolutePath(), true));
			write.println(new Double(win));
			if (write.checkError())
				System.out.println("Could not save the data!");
			write.close();
		}
		catch (IOException e) {
	   //   System.out.println("IOException trying to write: " + e);
		}
		finally {
			try {
				if (write != null)
					write.close();
			}
			catch (Exception e) {
	  //      System.out.println("Exception trying to close witer: " + e);
			}
		}
	}
}
