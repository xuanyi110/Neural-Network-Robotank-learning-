package JBot_NN;

public class States {
	public static int numHeading = 4;
	public static int numTargetDistance = 4;
	public static int numTargetBearing = 4;
	public static int numXPosition = 8;
	public static int numYPosition = 6;

	public static final int[][][][][] statesMap = new int[numHeading][numTargetDistance][numTargetBearing][numXPosition][numYPosition];
	public static int numStates = 0;
	public static void init(int[] f, int[] c) {
//		System.out.println("Im init-ing States!!!");
		if (f.length != 5 || c.length != 5) {
			System.out.println("Wrong number of state parameters!!!!!");
			return;
		}
		numHeading = c[0] - f[0] + 1;
		numTargetDistance = c[1] - f[1] + 1;
		numTargetBearing = c[2] - f[2] + 1;
		numXPosition = c[3] - f[3] + 1;
		numYPosition = c[4] - f[4] + 1;
		int count = 0;
		for (int i = 0; i < numHeading; i++)
			for (int j = 0; j < numTargetDistance; j++)
				for (int k = 0; k < numTargetBearing; k++)
					for (int l = 0; l < numXPosition; l++)
						for (int m = 0; m < numYPosition; m++)
							statesMap[i][j][k][l][m] = count++;
		numStates = count;
	}
	
	public static double getHeading(double arg){
		// 0 ~ 4
		return arg / Math.PI * 2 / 4;
	}

	public static double getTargetDistance(double arg){
		// 0 ~ 10
		if (arg > 400) return 1;
		return arg / 400;
	}

	public static double getTargetBearing(double arg){
		// 0 ~ 4
		return (arg + 180) / 90 / 4;
	}

	public static double getXPosition(double arg) {
		// 0 ~ 8
		return arg / 100 / 8;
	}

	public static double getYPosition(double arg) {
		// 0 ~ 6
		return arg / 100 / 6;
	}

//	public static int getHeading(double arg){
//		//4 
//		double unit = 2 * Math.PI / numHeading;
//		return (int)Math.floor(arg / unit);
//	}
//
//	public static int getTargetDistance(double arg){
//		//4 close, near, far, really far
//		int temp=(int)(arg/100);
//		if(temp > numTargetDistance - 1) temp = numTargetDistance - 1;
//		return temp;
//	}
//
//	public static int getTargetBearing(double arg){
//	//4 
//		double unit = 360.0 / numTargetBearing;
//		return (int)Math.floor((arg + 180) / unit);
//	}
//
//	public static int getXPosition(double arg) {
//		double unit = 800 / numXPosition;
//		return (int)Math.floor(arg / unit);
//	}
//
//	public static int getYPosition(double arg) {
//		double unit = 600 / numYPosition;
//		return (int)Math.floor(arg / unit);
//	}
}