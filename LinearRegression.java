import java.util.Arrays;

/**
 * Linear Regression implementation using Normal Equation method
 * Công thức: θ = (X^T * X)^(-1) * X^T * y
 */
public class LinearRegression {
    private double[] theta; // Hệ số hồi quy (weights)
    private int numFeatures;
    private String modelName;
    
    public LinearRegression(String modelName) {
        this.modelName = modelName;
    }
    
    /**
     * Train model using Normal Equation with Ridge Regularization
     * @param X Feature matrix (m x n) - m samples, n features
     * @param y Target vector (m x 1)
     */
    public void train(double[][] X, double[] y) {
        int m = X.length; // Số lượng samples
        int n = X[0].length; // Số lượng features
        this.numFeatures = n;
        
        // Thêm cột bias (intercept) - cột toàn số 1 vào đầu
        double[][] X_bias = addBiasColumn(X);
        
        // Tính X^T (transpose của X)
        double[][] X_transpose = transpose(X_bias);
        
        // Tính X^T * X
        double[][] XtX = multiply(X_transpose, X_bias);
        
        // Thêm Ridge regularization: (X^T * X + λI)
        // λ = 0.01 (small regularization to prevent singular matrix)
        double lambda = 0.01;
        for (int i = 0; i < XtX.length; i++) {
            XtX[i][i] += lambda;  // Add to diagonal
        }
        
        // Tính (X^T * X + λI)^(-1)
        double[][] XtX_inv = inverse(XtX);
        
        // Tính X^T * y
        double[] Xty = multiplyVector(X_transpose, y);
        
        // Tính θ = (X^T * X + λI)^(-1) * X^T * y
        this.theta = multiplyVector(XtX_inv, Xty);
        
        System.out.println("[" + modelName + "] Training completed with Ridge regularization!");
        // System.out.println("Theta (coefficients): " + Arrays.toString(theta));
    }
    
    /**
     * Predict target values for new data
     * @param X Feature matrix
     * @return Predicted values
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
     * Calculate R-squared (coefficient of determination)
     */
    public double calculateR2(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        // Tính mean của y
        double yMean = 0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= y.length;
        
        // Tính SS_tot và SS_res
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < y.length; i++) {
            ssTot += Math.pow(y[i] - yMean, 2);
            ssRes += Math.pow(y[i] - predictions[i], 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
    
    // ============ UTILITY METHODS ============
    
    /**
     * Thêm cột bias (cột toàn số 1) vào đầu matrix
     */
    private double[][] addBiasColumn(double[][] X) {
        int m = X.length;
        int n = X[0].length;
        double[][] X_bias = new double[m][n + 1];
        
        for (int i = 0; i < m; i++) {
            X_bias[i][0] = 1.0; // Bias column
            for (int j = 0; j < n; j++) {
                X_bias[i][j + 1] = X[i][j];
            }
        }
        return X_bias;
    }
    
    /**
     * Transpose matrix
     */
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
    
    /**
     * Nhân 2 ma trận
     */
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
    
    /**
     * Nhân ma trận với vector
     */
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
     * Tính ma trận nghịch đảo bằng Gauss-Jordan elimination
     */
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Tạo augmented matrix [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1.0;
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Tìm pivot
            double pivot = augmented[i][i];
            
            // Nếu pivot quá nhỏ, tìm row khác để swap
            if (Math.abs(pivot) < 1e-10) {
                int swapRow = -1;
                for (int k = i + 1; k < n; k++) {
                    if (Math.abs(augmented[k][i]) > 1e-10) {
                        swapRow = k;
                        break;
                    }
                }
                
                if (swapRow == -1) {
                    throw new RuntimeException("Matrix is singular and cannot be inverted. " +
                        "Consider using regularization or removing correlated features.");
                }
                
                // Swap rows
                double[] temp = augmented[i];
                augmented[i] = augmented[swapRow];
                augmented[swapRow] = temp;
                pivot = augmented[i][i];
            }
            
            // Chia hàng i cho pivot
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Khử các phần tử khác trong cột i
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Trích xuất ma trận nghịch đảo
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
        this.numFeatures = theta.length - 1; // Trừ bias term
    }
    
    public String getModelName() {
        return modelName;
    }
}