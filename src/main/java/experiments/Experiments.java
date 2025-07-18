package experiments;

import utils.Utils;

public class Experiments {

	public static void main(String[] args) throws Exception {

		String[] datasets = new String[] {
				"Scene",
				"GnegativePseAAC",
				"PlantPseAAC",
		};

		int[] numberLabels = new int[] {
				6,
				8,
				12,
			    
		};
		
		String[] algorithms = new String[] {
				"moa.classifiers.multilabel.meta.BRALDDS",
		};

		String[] algorithmNames = new String[] {
				"MLALDDS",
				
		};
		
		// Executables
		System.out.println("===== Executables =====");
		for(int dat = 0; dat < datasets.length; dat++) {
			for(int alg = 0; alg < algorithmNames.length; alg++)
			{
				String memory = "-XX:ParallelGCThreads= 12 -Xms16g -Xmx128g";

				System.out.println("java " + memory + " -javaagent:sizeofag-1.0.4.jar -cp MLALDDS-1.0-MLALDDS-jar-with-dependencies.jar "
						+ "moa.DoTask EvaluatePrequentialMultiLabel "
						+ " -e \"(PrequentialMultiLabelPerformanceEvaluator)\""
						+ " -s \"(MultiTargetArffFileStream -c " + numberLabels[dat] + " -f datasets/" + datasets[dat] + ".arff)\"" 
						+ " -l \"(" + algorithms[alg] + ")\""
						+ " -f 100"
						+ " -d results/" + algorithmNames[alg] + "-" + datasets[dat] + ".csv");
			}
		}
		
		// Show metrics for results
		System.out.println("===== Results =====");
		
		Utils.metric("Subset Accuracy", "averaged", "results", algorithmNames, datasets);
		Utils.metric("Hamming Score", "averaged", "results", algorithmNames, datasets);
		Utils.metric("Example-Based Accuracy", "averaged", "results", algorithmNames, datasets);
		Utils.metric("Example-Based F-Measure", "averaged", "results", algorithmNames, datasets);
		
		Utils.metric("evaluation time (cpu seconds)", "last", "results", algorithmNames, datasets);
		Utils.metric("model cost (RAM-Hours)", "averaged", "results", algorithmNames, datasets);
	}
}