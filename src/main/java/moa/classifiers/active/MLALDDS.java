package moa.classifiers.active;

/*
 *   Multi-Label Active Learning for Drifting Data Streams (MLALDDS)
 *    Copyright (C) 
 *    @author Reza Rahimian
 *   
 */

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.lazy.SAMkNNplus;
import moa.core.Measurement;
import moa.core.DoubleVector;
import moa.core.Utils;
import moa.options.ClassOption;


public class MLALDDS extends AbstractClassifier implements ALClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Active learning for Multi-label drifting data streams";
    }
    
    // Options
    
    // Self-adjusting kNN algorithm (SAMkNN+), as the base classifier within a BR architecture
    public ClassOption coreLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", SAMkNNplus.class, "moa.classifiers.lazy.SAMkNNplus"); 
    public FloatOption initialPhaseInstances = new FloatOption("numInstancesInit",
            'n', "Number of instances at beginning without active learning.",
            0.0, 0.00, Integer.MAX_VALUE);

    public FloatOption labelingBudgetRatio = new FloatOption("labelingBudget",
            'g', "Budget to use for uncertainty AL.",
            0.10, 0.0, 1.0);

    public FloatOption uncertaintySmoothingFactor = new FloatOption("uncertaintySmoothing",
            'b', "Smoothing parameter (b) for uncertainty-based query strategy.",
            0.10, 0.0, 1.0);

    public IntOption slidingWindowSize = new IntOption("memoryWindowSize",
            'm', "Number of instances to store",
            300, 1, Integer.MAX_VALUE);

    // Core fields
    public double smoothedBudget;
    public boolean shouldLabel;
    public int queryFlag;
    public double labelUpdateEstimate;

    public Classifier mainModel;
    protected Classifier shadowModel = null;

    private double latestPosterior;
    public int labelCostCounter;
    public int recentLabeling;
    public double maxPosteriorSeen;
    public double modelAccuracy;
    public int iterationIndex;

    protected long totalInstances;
    protected long firstWarningInstance;
    protected boolean warningFlag;
    protected ADWIN driftDetector;
    protected int driftCount = 0;
    protected int warningCount = 0;

    /** Compute posterior probability */
    private double estimatePosterior(double[] predictionVector) {
        if (predictionVector.length > 1) {
            DoubleVector voteVector = new DoubleVector(predictionVector);
            if (voteVector.sumOfValues() > 0.0) {
                voteVector.normalize();
            }
            predictionVector = voteVector.getArrayRef();
            latestPosterior = predictionVector[Utils.maxIndex(predictionVector)];
        } else {
            latestPosterior = 0;
        }
        return latestPosterior;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.mainModel.getVotesForInstance(inst);
    }

    /** Initialization of learning process */
    @Override
    public void resetLearningImpl() {
        this.mainModel = ((Classifier) getPreparedClassOption(this.coreLearnerOption)).copy();
        this.mainModel.resetLearning();
        this.shadowModel = null;
        this.warningFlag = false;
        this.firstWarningInstance = 0;
        this.labelCostCounter = 0;
        this.recentLabeling = 0;
        this.iterationIndex = 0;
        this.modelAccuracy = 0;
        this.labelUpdateEstimate = 0;
        this.queryFlag = 0;
        this.driftDetector = new ADWIN();
    }

    /** Main training logic with budget and drift handling */
    @Override
    public void trainOnInstanceImpl(Instance inst) {

        this.iterationIndex++;
        double budgetRatio;

        double lambda = 1.0 - (1.0 / this.slidingWindowSize.getValue());
        this.queryFlag = 0;
        boolean warningDetected = false;
        boolean driftDetected = false;

        if (this.iterationIndex <= this.initialPhaseInstances.getValue()) {
            budgetRatio = 0;
            this.queryFlag = 1;
            this.mainModel.trainOnInstance(inst);
            this.labelCostCounter++;
            this.labelUpdateEstimate++;
            return;
        } else {
            budgetRatio = (this.labelCostCounter - this.initialPhaseInstances.getValue()) /
                    ((double) this.iterationIndex - this.initialPhaseInstances.getValue());
        }

        double oldEstimation = this.driftDetector.getEstimation();
        Prediction pred = this.mainModel.getPredictionForInstance(inst);

        for (int i = 0; i < inst.numOutputAttributes(); i++) {
            if (pred.getVotes(i) != null) {
                boolean isCorrect = Utils.maxIndex(pred.getVotes(i)) == (int) inst.classValue(i);
                boolean changed = this.driftDetector.setInput(isCorrect ? 0 : 1);
                if (changed && this.driftDetector.getEstimation() > oldEstimation) {
                    warningDetected = true;
                }
                if (this.driftDetector.getChange()) {
                    driftDetected = true;
                }
            }
        }

        if (budgetRatio < this.labelingBudgetRatio.getValue()) {

            this.totalInstances++;

            if (warningDetected && this.shadowModel == null) {
                this.mainModel.resetLearning();
                this.shadowModel = this.mainModel.copy();
                this.shadowModel.resetLearning();
                this.driftDetector = new ADWIN();
                this.warningCount++;
                this.warningFlag = true;
                this.firstWarningInstance = totalInstances;
            }

            if (driftDetected && this.shadowModel != null) {
                this.mainModel = this.shadowModel.copy();
                this.shadowModel = null;
                this.driftDetector = new ADWIN();
                this.warningFlag = false;
                this.driftCount++;
            }

            this.maxPosteriorSeen = estimatePosterior(this.mainModel.getVotesForInstance(inst));
            double p = Math.abs(maxPosteriorSeen - 1.0 / inst.numClasses());
            double queryThreshold = this.uncertaintySmoothingFactor.getValue() /
                    (this.uncertaintySmoothingFactor.getValue() + p);

            if (this.classifierRandom.nextDouble() < queryThreshold) {
                this.mainModel.trainOnInstance(inst);
                if (this.shadowModel != null) {
                    this.shadowModel.trainOnInstance(inst);
                }
                this.queryFlag = 1;
                this.labelUpdateEstimate++;
                this.recentLabeling++;
            }
        }

        this.labelCostCounter += this.queryFlag;
        this.labelUpdateEstimate = this.labelUpdateEstimate * lambda + this.queryFlag;
        budgetRatio = this.labelUpdateEstimate / slidingWindowSize.getValue();
        this.totalInstances++;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] innerMeasurements = ((AbstractMultiLabelLearner) this.mainModel).getModelMeasurements();
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier) this.mainModel).getModelDescription(out, indent);
    }

    @Override
    public int getLastLabelAcqReport() {
        int temp = this.recentLabeling;
        this.recentLabeling = 0;
        return temp;
    }

    @Override
    public void setModelContext(InstancesHeader ih) {
        super.setModelContext(ih);
        mainModel.setModelContext(ih);
    }
} // End of MLALDDS class
