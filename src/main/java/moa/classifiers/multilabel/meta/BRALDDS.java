package moa.classifiers.multilabel.meta;

/*
 * Binary relevance architecture for Multi-Label Active Learning for Drifting Data Streams
 */

import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.Classifier;
import moa.classifiers.MultiLabelClassifier;
import moa.classifiers.active.MLALDDS;
import moa.core.FastVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.InstanceStream;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;


public class BRALDDS extends AbstractMultiLabelLearner implements MultiLabelClassifier{

	public IntOption seedInitializer = new IntOption("randomSeedOption",
			'r', "randomSeedOption",
			1, Integer.MIN_VALUE, Integer.MAX_VALUE);

	public BRALDDS() {
		super.randomSeedOption = seedInitializer;
		initializeStructure();
	}

	protected void initializeStructure() {
		coreModelOption = new ClassOption("baseLearner", 'l',
				"Classifier to train.", Classifier.class, MLALDDS.class.getName());
	}

	private static final long serialVersionUID = 1L;
	
	@Override
	    public String getPurposeString() {
	        return "Multi-Label Active Learning for Drifting Data Streams";
	    }

	// Options
	public ClassOption coreModelOption;

	protected Classifier[] binaryEnsemble;

	protected boolean modelInitialized = false;

	@Override
	public void resetLearningImpl() {
		this.modelInitialized = false;
		if (binaryEnsemble != null) {
			for (int i = 0; i < binaryEnsemble.length; i++) {
				binaryEnsemble[i].resetLearning();
			}
		}
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance streamInstance) {
		if (!this.modelInitialized) {
			this.binaryEnsemble = new Classifier[streamInstance.numberOutputTargets()];
			Classifier coreClassifier = (Classifier) getPreparedClassOption(this.coreModelOption);

			if (coreClassifier.isRandomizable())
				coreClassifier.setRandomSeed(this.randomSeed);
			coreClassifier.resetLearning();

			for (int i = 0; i < this.binaryEnsemble.length; i++) {
				this.binaryEnsemble[i] = coreClassifier.copy();
			}
			this.modelInitialized = true;
		}
		for (int i = 0; i < this.binaryEnsemble.length; i++) {
			Instance newInst = convertToBinaryInstance(streamInstance, i);
			this.binaryEnsemble[i].trainOnInstance(newInst);
		}
	}

	protected InstancesHeader[] ensembleHeaders;

	protected Instance convertToBinaryInstance(MultiLabelInstance srcInstance, int labelIndex) {
		if (ensembleHeaders == null) {
			this.ensembleHeaders = new InstancesHeader[this.binaryEnsemble.length];
		}
		if (ensembleHeaders[labelIndex] == null) {
			FastVector<Attribute> inputAttributes = new FastVector<>();
			for (int j = 0; j < srcInstance.numInputAttributes(); j++) {
				inputAttributes.addElement(srcInstance.inputAttribute(j));
			}
			inputAttributes.addElement(srcInstance.outputAttribute(labelIndex));
			this.ensembleHeaders[labelIndex] = new InstancesHeader(new Instances(
					getCLICreationString(InstanceStream.class), inputAttributes, 0));
			this.ensembleHeaders[labelIndex].setClassIndex(inputAttributes.size() - 1);
			this.binaryEnsemble[labelIndex].setModelContext(this.ensembleHeaders[labelIndex]);
		}

		int inputCount = this.ensembleHeaders[labelIndex].numInputAttributes();
		double[] values = new double[inputCount + 1];
		for (int k = 0; k < inputCount; k++) {
			values[k] = srcInstance.valueInputAttribute(k);
		}
		Instance transformed = new DenseInstance(1.0, values);
		transformed.setDataset(ensembleHeaders[labelIndex]);
		transformed.setClassValue(srcInstance.valueOutputAttribute(labelIndex));
		return transformed;
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// Optional model description logic
	}

	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance multiInst) {
		Prediction labelPrediction = null;

		if (this.modelInitialized) {
			labelPrediction = new MultiLabelPrediction(binaryEnsemble.length);
			for (int i = 0; i < this.binaryEnsemble.length; i++) {
				Instance instBinary = convertToBinaryInstance(multiInst, i);
				double[] voteResults = this.binaryEnsemble[i].getVotesForInstance(instBinary);

				if (instBinary.classAttribute().isNumeric()) {
					labelPrediction.setVote(i, 0, voteResults[0]);
				} else {
					double[] normalizedVotes = new double[voteResults.length];
					double total = 0;
					for (int j = 0; j < voteResults.length; j++) {
						normalizedVotes[j] = voteResults[j];
						total += voteResults[j];
					}
					for (int j = 0; j < voteResults.length; j++) {
						normalizedVotes[j] /= total;
					}
					labelPrediction.setVotes(i, normalizedVotes);
				}
			}
		}
		return labelPrediction;
	}

	@Override
	public int measureByteSize() {
		int size = 0;
		if (this.binaryEnsemble != null) {
			for (Classifier model : this.binaryEnsemble) {
				if (model != null) {
					try {
						ByteArrayOutputStream outStream = new ByteArrayOutputStream();
						ObjectOutputStream objStream = new ObjectOutputStream(outStream);
						objStream.writeObject(model);
						objStream.close();
						size += outStream.size();
					} catch (Exception e) {
						System.err.println("Error measuring model size: " + e.getMessage());
						size += 1024;
					}
				}
			}
		} else {
			size = 100 * 1024;
		}
		return size;
	}
}

