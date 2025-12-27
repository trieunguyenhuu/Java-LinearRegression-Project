import java.io.*;

/**
 * Utility class to save and load trained models
 */
public class ModelSerializer {
    
    /**
     * Save trained model to file
     */
    public static void saveModel(LinearRegression model, String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(model.getTheta());
            System.out.println("[✓] Model saved: " + filename);
        }
    }
    
    /**
     * Load trained model from file
     */
    public static LinearRegression loadModel(String modelName, String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            double[] theta = (double[]) ois.readObject();
            LinearRegression model = new LinearRegression(modelName);
            model.setTheta(theta);
            System.out.println("[✓] Model loaded: " + filename);
            return model;
        }
    }
    
    /**
     * Check if model files exist
     */
    public static boolean modelsExist() {
        File model1 = new File("model_total_spend.dat");
        File model2 = new File("model_frequency.dat");
        File model3 = new File("model_entertainment.dat");
        
        return model1.exists() && model2.exists() && model3.exists();
    }
    
    /**
     * Check if specific model files exist (for focused approach)
     */
    public static boolean modelsExist(String... modelFiles) {
        for (String filename : modelFiles) {
            File file = new File(filename);
            if (!file.exists()) {
                return false;
            }
        }
        return true;
    }
}