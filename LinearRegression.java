import java.util.Arrays;

/**
 * Linear Regression with Ridge Regularization
 * Formula: θ = (X^T * X + λI)^(-1) * X^T * y
 * Ridge regularization prevents singular matrix issues
 */
public class LinearRegression {
    private double[] theta;
    private int numFeatures;
    private String modelName;
    private double lambda; // Regularization parameter
    
    public LinearRegression(String modelName) {
        this(modelName, 0.01); // Default lambda = 0.01
    }
    
    public LinearRegression(String modelName, double lambda) {
        this.modelName = modelName;
        this.lambda = lambda;
    }
    
    /**
     * Train model using Ridge Regression (Normal Equation with regularization)
     * θ = (X^T * X + λI)^(-1) * X^T * y
     */
    public void train(double[][] X, double[] y) {
        int m = X.length;
        int n = X[0].length;
        this.numFeatures = n;
        
        System.out.println("[Training] Samples: " + m + ", Features: " + n);
        System.out.println("[Training] Lambda (regularization): " + lambda);
        
        // Add bias column
        double[][] X_bias = addBiasColumn(X);
        
        // Calculate X^T
        double[][] X_transpose = transpose(X_bias);
        
        // Calculate X^T * X
        double[][] XtX = multiply(X_transpose, X_bias);
        
        // Add regularization: X^T * X + λI
        // Note: Don't regularize bias term (first row/col)
        int dim = XtX.length;
        for (int i = 1; i < dim; i++) { // Start from 1 to skip bias
            XtX[i][i] += lambda;
        }
        
        // Calculate (X^T * X + λI)^(-1)
        double[][] XtX_inv;
        try {
            XtX_inv = inverse(XtX);
        } catch (RuntimeException e) {
            System.err.println("Matrix still singular even with regularization!");
            System.err.println("Trying with higher lambda...");
            
            // Try with higher lambda
            for (int i = 1; i < dim; i++) {
                XtX[i][i] += lambda * 99; // Total lambda = 100 * original
            }
            XtX_inv = inverse(XtX);
        }
        
        // Calculate X^T * y
        double[] Xty = multiplyVector(X_transpose, y);
        
        // Calculate θ = (X^T * X + λI)^(-1) * X^T * y
        this.theta = multiplyVector(XtX_inv, Xty);
        
        System.out.println("[" + modelName + "] Training completed!");
        System.out.println("Theta (first 5): " + Arrays.toString(Arrays.copyOf(theta, Math.min(5, theta.length))));
    }
    
    /**
     * Predict target values
     */
    public double[] predict(double[][] X) {
        int m = X.length;
        double[][] X_bias = addBiasColumn(X);
        double[] predictions = new double[m];
        
        for (int i = 0; i < m; i++) {
            predictions[i] = 0;
            for (int j = 0; j < theta.length; j++) {
                predictions[i] += theta[j] * X_bias[i][j];
            }
        }
        
        return predictions;
    }
    
    /**
     * Predict single sample
     */
    public double predictSingle(double[] features) {
        double prediction = theta[0]; // bias
        for (int i = 0; i < features.length; i++) {
            prediction += theta[i + 1] * features[i];
        }
        return prediction;
    }
    
    /**
     * Calculate Mean Squared Error
     */
    public double calculateMSE(double[][] X, double[] y) {
        double[] predictions = predict(X);
        double mse = 0;
        for (int i = 0; i < y.length; i++) {
            double error = predictions[i] - y[i];
            mse += error * error;
        }
        return mse / y.length;
    }
    
    /**
     * Calculate R-squared
     */
    public double calculateR2(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        double yMean = 0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= y.length;
        
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < y.length; i++) {
            ssTot += Math.pow(y[i] - yMean, 2);
            ssRes += Math.pow(y[i] - predictions[i], 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
    
    // ============ UTILITY METHODS ============
    
    private double[][] addBiasColumn(double[][] X) {
        int m = X.length;
        int n = X[0].length;
        double[][] X_bias = new double[m][n + 1];
        
        for (int i = 0; i < m; i++) {
            X_bias[i][0] = 1.0;
            for (int j = 0; j < n; j++) {
                X_bias[i][j + 1] = X[i][j];
            }
        }
        return X_bias;
    }
    
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    private double[][] multiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                result[i][j] = 0;
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    private double[] multiplyVector(double[][] A, double[] b) {
        int rows = A.length;
        int cols = A[0].length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            result[i] = 0;
            for (int j = 0; j < cols; j++) {
                result[i] += A[i][j] * b[j];
            }
        }
        return result;
    }
    
    /**
     * Improved matrix inversion with better numerical stability
     */
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1.0;
        }
        
        // Gauss-Jordan elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Find pivot (largest element in column)
            int maxRow = i;
            double maxVal = Math.abs(augmented[i][i]);
            for (int k = i + 1; k < n; k++) {
                double val = Math.abs(augmented[k][i]);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[maxRow];
                augmented[maxRow] = temp;
            }
            
            // Check for singular matrix
            double pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new RuntimeException("Matrix is singular and cannot be inverted (pivot too small: " + pivot + ")");
            }
            
            // Divide row by pivot
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate other rows
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = augmented[i][n + j];
            }
        }
        
        return result;
    }
    
    public double[] getTheta() {
        return theta;
    }
    
    public void setTheta(double[] theta) {
        this.theta = theta;
        this.numFeatures = theta.length - 1;
    }
    
    public String getModelName() {
        return modelName;
    }
    
    public void setLambda(double lambda) {
        this.lambda = lambda;
    }
}