import java.io.IOException;
import java.util.Scanner;

/**
 * Spending Prediction System - Time Series with Confidence Metrics
 * Version 6.0: Train/Val/Test split + Confidence scores + Real value conversion
 */
public class SpendingPrediction {
    
    private static LinearRegression model1, model2, model3;
    private static DataLoader.Dataset fullDataset1, fullDataset2, fullDataset3;
    
    // Performance metrics for confidence calculation
    private static PerformanceMetrics metrics1, metrics2, metrics3;
    
    // Scaler for denormalization
    private static SimpleScalerInfo scalerInfo;
    
    // Model files
    private static final String MODEL1_FILE = "model_total_spend_ts.dat";
    private static final String MODEL2_FILE = "model_frequency_ts.dat";
    private static final String MODEL3_FILE = "model_entertainment_ts.dat";
    
    // Data files
    private static final String DATA1_FILE = "customer_spending_cleaned_Y1_Total_Spend.csv";
    private static final String DATA2_FILE = "customer_spending_cleaned_Y2_Frequency.csv";
    private static final String DATA3_FILE = "customer_spending_cleaned_Y3_Entertainment.csv";
    
    // Target prediction month
    private static final int TARGET_YEAR = 2025;
    private static final int TARGET_MONTH = 12;
    
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(80));
            System.out.println("CUSTOMER SPENDING PREDICTION SYSTEM v6.0");
            System.out.println("Time Series with Train/Val/Test + Confidence Metrics");
            System.out.println("=".repeat(80));
            System.out.println();
            
            // Initialize scaler
            scalerInfo = new SimpleScalerInfo();
            
            // Load data
            System.out.println("[STEP 1] Loading time series datasets... (Đang tải dữ liệu chuỗi thời gian...)");
            System.out.println("=".repeat(80));
            fullDataset1 = DataLoader.loadFromCSV(DATA1_FILE, "Total_Monthly_Spend");
            fullDataset2 = DataLoader.loadFromCSV(DATA2_FILE, "Frequency_Total");
            fullDataset3 = DataLoader.loadFromCSV(DATA3_FILE, "Amount_Entertainment");
            System.out.println("=".repeat(80));
            System.out.println();
            
            // Check if models exist
            if (ModelSerializer.modelsExist(MODEL1_FILE, MODEL2_FILE, MODEL3_FILE)) {
                System.out.println("=".repeat(80));
                System.out.println("✓ Phát hiện models đã được train trước đó!");
                System.out.println("=".repeat(80));
                System.out.println();
                
                Scanner scanner = new Scanner(System.in);
                System.out.print("Bạn muốn:\n");
                System.out.print("  [1] Load models có sẵn (nhanh)\n");
                System.out.print("  [2] Train lại từ đầu\n");
                System.out.print("Chọn (1/2): ");
                
                String choice = scanner.nextLine().trim();
                System.out.println();
                
                if (choice.equals("1")) {
                    loadModels();
                    evaluateAllModels();
                } else {
                    trainAndSaveModels();
                }
            } else {
                System.out.println("⚠ Chưa có models được train. Đang bắt đầu training...\n");
                trainAndSaveModels();
            }
            
            // Interactive prediction
            interactivePrediction();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void trainAndSaveModels() throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("TRAINING MODELS WITH TRAIN/VAL/TEST SPLIT");
        System.out.println("(ĐÀO TẠO MODELS VỚI PHÂN CHIA TRAIN/VAL/TEST)");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // Split datasets: 60% train, 20% val, 20% test
        System.out.println("[STEP 2] Splitting datasets (60/20/20)... (Đang phân chia datasets...)");
        System.out.println();
        
        System.out.println("Dataset 1: Total_Monthly_Spend");
        DataLoader.Dataset[] splits1 = DataLoader.splitTrainValTest(fullDataset1, 0.6, 0.2);
        System.out.println();
        
        System.out.println("Dataset 2: Frequency_Total");
        DataLoader.Dataset[] splits2 = DataLoader.splitTrainValTest(fullDataset2, 0.6, 0.2);
        System.out.println();
        
        System.out.println("Dataset 3: Amount_Entertainment");
        DataLoader.Dataset[] splits3 = DataLoader.splitTrainValTest(fullDataset3, 0.6, 0.2);
        System.out.println();
        
        DataLoader.Dataset train1 = splits1[0], val1 = splits1[1], test1 = splits1[2];
        DataLoader.Dataset train2 = splits2[0], val2 = splits2[1], test2 = splits2[2];
        DataLoader.Dataset train3 = splits3[0], val3 = splits3[1], test3 = splits3[2];
        
        // Train models
        System.out.println("[STEP 3] Training models... (Đang đào tạo models...)");
        System.out.println("=".repeat(80));
        
        System.out.println("\n>>> Model 1: Total Monthly Spend (Tổng chi tiêu hàng tháng) <<<");
        System.out.println("Features: " + train1.X[0].length);
        model1 = new LinearRegression("Total_Monthly_Spend");
        model1.train(train1.X, train1.getYPrimitive());
        
        System.out.println("\n>>> Model 2: Transaction Frequency (Tần suất giao dịch) <<<");
        System.out.println("Features: " + train2.X[0].length);
        model2 = new LinearRegression("Frequency_Total");
        model2.train(train2.X, train2.getYPrimitive());
        
        System.out.println("\n>>> Model 3: Entertainment Spending (Chi tiêu giải trí) <<<");
        System.out.println("Features: " + train3.X[0].length);
        model3 = new LinearRegression("Amount_Entertainment");
        model3.train(train3.X, train3.getYPrimitive());
        
        System.out.println();
        System.out.println("=".repeat(80));
        
        // Evaluate on all sets
        System.out.println("\n[STEP 4] Evaluating models... (Đang đánh giá models...)");
        System.out.println("=".repeat(80));
        
        metrics1 = evaluateModelOnAllSets(model1, train1, val1, test1);
        System.out.println();
        
        metrics2 = evaluateModelOnAllSets(model2, train2, val2, test2);
        System.out.println();
        
        metrics3 = evaluateModelOnAllSets(model3, train3, val3, test3);
        System.out.println();
        
        System.out.println("=".repeat(80));
        
        // Save models
        System.out.println("\n[STEP 5] Saving models... (Đang lưu models...)");
        System.out.println("-".repeat(80));
        ModelSerializer.saveModel(model1, MODEL1_FILE);
        ModelSerializer.saveModel(model2, MODEL2_FILE);
        ModelSerializer.saveModel(model3, MODEL3_FILE);
        System.out.println("-".repeat(80));
        System.out.println("✓ Models saved! (Đã lưu models!)");
        System.out.println();
        
        System.out.println("=".repeat(80));
        System.out.println("TRAINING COMPLETED! (ĐÀO TẠO HOÀN TẤT!)");
        System.out.println("=".repeat(80));
        System.out.println();
    }
    
    private static void loadModels() throws IOException, ClassNotFoundException {
        System.out.println("=".repeat(80));
        System.out.println("LOADING PRE-TRAINED MODELS (ĐANG TẢI MODELS ĐÃ TRAIN)");
        System.out.println("=".repeat(80));
        System.out.println();
        
        model1 = ModelSerializer.loadModel("Total_Monthly_Spend", MODEL1_FILE);
        model2 = ModelSerializer.loadModel("Frequency_Total", MODEL2_FILE);
        model3 = ModelSerializer.loadModel("Amount_Entertainment", MODEL3_FILE);
        
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("✓ MODELS LOADED! (ĐÃ TẢI MODELS!)");
        System.out.println("=".repeat(80));
        System.out.println();
    }
    
    private static void evaluateAllModels() {
        System.out.println("=".repeat(80));
        System.out.println("EVALUATING LOADED MODELS (ĐÁNH GIÁ MODELS ĐÃ TẢI)");
        System.out.println("=".repeat(80));
        System.out.println();
        
        DataLoader.Dataset[] splits1 = DataLoader.splitTrainValTest(fullDataset1, 0.6, 0.2);
        DataLoader.Dataset[] splits2 = DataLoader.splitTrainValTest(fullDataset2, 0.6, 0.2);
        DataLoader.Dataset[] splits3 = DataLoader.splitTrainValTest(fullDataset3, 0.6, 0.2);
        
        metrics1 = evaluateModelOnAllSets(model1, splits1[0], splits1[1], splits1[2]);
        System.out.println();
        
        metrics2 = evaluateModelOnAllSets(model2, splits2[0], splits2[1], splits2[2]);
        System.out.println();
        
        metrics3 = evaluateModelOnAllSets(model3, splits3[0], splits3[1], splits3[2]);
        System.out.println();
        
        System.out.println("=".repeat(80));
    }
    
    private static PerformanceMetrics evaluateModelOnAllSets(LinearRegression model,
                                                             DataLoader.Dataset train,
                                                             DataLoader.Dataset val,
                                                             DataLoader.Dataset test) {
        System.out.println(">>> " + model.getModelName() + " <<<");
        
        // Train performance
        double[] y_train = train.getYPrimitive();
        double trainMSE = model.calculateMSE(train.X, y_train);
        double trainRMSE = Math.sqrt(trainMSE);
        double trainR2 = model.calculateR2(train.X, y_train);
        double trainMAPE = calculateMAPE(model, train.X, y_train);
        
        // Validation performance
        double[] y_val = val.getYPrimitive();
        double valMSE = model.calculateMSE(val.X, y_val);
        double valRMSE = Math.sqrt(valMSE);
        double valR2 = model.calculateR2(val.X, y_val);
        double valMAPE = calculateMAPE(model, val.X, y_val);
        
        // Test performance
        double[] y_test = test.getYPrimitive();
        double testMSE = model.calculateMSE(test.X, y_test);
        double testRMSE = Math.sqrt(testMSE);
        double testR2 = model.calculateR2(test.X, y_test);
        double testMAPE = calculateMAPE(model, test.X, y_test);
        
        System.out.println("\nTrain Performance (Hiệu suất Train):");
        System.out.printf("  MSE:  %.6f\n", trainMSE);
        System.out.printf("  RMSE: %.6f\n", trainRMSE);
        System.out.printf("  R²:   %.4f (%.2f%%)\n", trainR2, trainR2 * 100);
        System.out.printf("  MAPE: %.2f%%\n", trainMAPE);
        
        System.out.println("Validation Performance (Hiệu suất Validation):");
        System.out.printf("  MSE:  %.6f\n", valMSE);
        System.out.printf("  RMSE: %.6f\n", valRMSE);
        System.out.printf("  R²:   %.4f (%.2f%%)\n", valR2, valR2 * 100);
        System.out.printf("  MAPE: %.2f%%\n", valMAPE);
        
        System.out.println("Test Performance (Hiệu suất Test):");
        System.out.printf("  MSE:  %.6f\n", testMSE);
        System.out.printf("  RMSE: %.6f\n", testRMSE);
        System.out.printf("  R²:   %.4f (%.2f%%)\n", testR2, testR2 * 100);
        System.out.printf("  MAPE: %.2f%%\n", testMAPE);
        
        return new PerformanceMetrics(trainR2, trainMAPE, valR2, valMAPE, testR2, testMAPE);
    }
    
    private static double calculateMAPE(LinearRegression model, double[][] X, double[] y) {
        double[] predictions = model.predict(X);
        double mape = 0;
        int validCount = 0;
        
        for (int i = 0; i < y.length; i++) {
            if (Math.abs(y[i]) > 1e-6) {
                mape += Math.abs((y[i] - predictions[i]) / y[i]);
                validCount++;
            }
        }
        
        return validCount > 0 ? (mape / validCount) * 100 : 0;
    }
    
    private static void interactivePrediction() {
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.println();
            System.out.println("=".repeat(80));
            System.out.println("PREDICTION FOR DECEMBER 2025 (DỰ BÁO THÁNG 12/2025)");
            System.out.println("=".repeat(80));
            
            System.out.print("Nhập Account_Key (hoặc 'exit' để thoát): ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) {
                System.out.println("\nĐã thoát chương trình. Tạm biệt!");
                break;
            }
            
            try {
                int accountKey = Integer.parseInt(input);
                predictForAccount(accountKey);
                
            } catch (NumberFormatException e) {
                System.out.println("❌ Lỗi: Vui lòng nhập số nguyên hợp lệ!");
            }
        }
        
        scanner.close();
    }
    
    private static void predictForAccount(int accountKey) {
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("KẾT QUẢ DỰ BÁO (PREDICTION RESULT)");
        System.out.println("=".repeat(80));
        System.out.println("Account_Key: " + accountKey);
        System.out.println("Tháng dự báo (Target month): " + TARGET_YEAR + "/" + TARGET_MONTH);
        System.out.println("=".repeat(80));
        
        boolean found = false;
        
        // Find data for this account
        // Get prediction features for December 2025
        double[] features1 = findFeatures(fullDataset1, accountKey, TARGET_YEAR, TARGET_MONTH);
        double[] features2 = findFeatures(fullDataset2, accountKey, TARGET_YEAR, TARGET_MONTH);
        double[] features3 = findFeatures(fullDataset3, accountKey, TARGET_YEAR, TARGET_MONTH);
        
        // Get previous month data (November 2025)
        double prevValue1 = findPreviousValue(fullDataset1, accountKey, TARGET_YEAR, TARGET_MONTH - 1);
        double prevValue2 = findPreviousValue(fullDataset2, accountKey, TARGET_YEAR, TARGET_MONTH - 1);
        double prevValue3 = findPreviousValue(fullDataset3, accountKey, TARGET_YEAR, TARGET_MONTH - 1);
        
        if (features1 != null) {
            found = true;
            double predNorm1 = model1.predictSingle(features1);
            double predReal1 = scalerInfo.denormalize("Total_Monthly_Spend", predNorm1);
            double prevReal1 = scalerInfo.denormalize("Total_Monthly_Spend", prevValue1);
            
            System.out.println("\n[1] TOTAL MONTHLY SPEND (TỔNG CHI TIÊU HÀNG THÁNG)");
            System.out.println("-".repeat(80));
            System.out.printf("  Tháng trước (11/2025):     %,.0f VND\n", prevReal1);
            System.out.printf("  Dự báo (12/2025):          %,.0f VND\n", predReal1);
            
            double change1 = predReal1 - prevReal1;
            double changePct1 = prevReal1 > 0 ? (change1 / prevReal1) * 100 : 0;
            String trend1 = change1 >= 0 ? "Tăng ↑" : "Giảm ↓";
            System.out.printf("  Thay đổi:                  %s %,.0f VND (%.2f%%)\n", 
                             trend1, Math.abs(change1), Math.abs(changePct1));
            
            printConfidence("Total Spend", metrics1);
        }
        
        if (features2 != null) {
            found = true;
            double predNorm2 = model2.predictSingle(features2);
            double predReal2 = scalerInfo.denormalize("Frequency_Total", predNorm2);
            double prevReal2 = scalerInfo.denormalize("Frequency_Total", prevValue2);
            
            System.out.println("\n[2] TRANSACTION FREQUENCY (TẦN SUẤT GIAO DỊCH)");
            System.out.println("-".repeat(80));
            System.out.printf("  Tháng trước (11/2025):     %.0f lần\n", prevReal2);
            System.out.printf("  Dự báo (12/2025):          %.0f lần\n", predReal2);
            
            double change2 = predReal2 - prevReal2;
            double changePct2 = prevReal2 > 0 ? (change2 / prevReal2) * 100 : 0;
            String trend2 = change2 >= 0 ? "Tăng ↑" : "Giảm ↓";
            System.out.printf("  Thay đổi:                  %s %.0f lần (%.2f%%)\n", 
                             trend2, Math.abs(change2), Math.abs(changePct2));
            
            printConfidence("Frequency", metrics2);
        }
        
        if (features3 != null) {
            found = true;
            double predNorm3 = model3.predictSingle(features3);
            double predReal3 = scalerInfo.denormalize("Amount_Entertainment", predNorm3);
            double prevReal3 = scalerInfo.denormalize("Amount_Entertainment", prevValue3);
            
            System.out.println("\n[3] ENTERTAINMENT SPENDING (CHI TIÊU GIẢI TRÍ)");
            System.out.println("-".repeat(80));
            System.out.printf("  Tháng trước (11/2025):     %,.0f VND\n", prevReal3);
            System.out.printf("  Dự báo (12/2025):          %,.0f VND\n", predReal3);
            
            double change3 = predReal3 - prevReal3;
            double changePct3 = prevReal3 > 0 ? (change3 / prevReal3) * 100 : 0;
            String trend3 = change3 >= 0 ? "Tăng ↑" : "Giảm ↓";
            System.out.printf("  Thay đổi:                  %s %,.0f VND (%.2f%%)\n", 
                             trend3, Math.abs(change3), Math.abs(changePct3));
            
            printConfidence("Entertainment", metrics3);
        }
        
        if (!found) {
            System.out.println("\n❌ Không tìm thấy dữ liệu cho Account_Key=" + accountKey);
        }
        
        System.out.println("\n" + "=".repeat(80));
    }
    
    private static double[] findFeatures(DataLoader.Dataset dataset, int accountKey, int year, int month) {
        for (int i = 0; i < dataset.accountKeys.length; i++) {
            if (dataset.accountKeys[i] == accountKey && 
                dataset.years[i] == year && 
                dataset.months[i] == month) {
                return dataset.X[i];
            }
        }
        return null;
    }
    
    private static double findPreviousValue(DataLoader.Dataset dataset, int accountKey, int year, int month) {
        for (int i = 0; i < dataset.accountKeys.length; i++) {
            if (dataset.accountKeys[i] == accountKey && 
                dataset.years[i] == year && 
                dataset.months[i] == month) {
                return dataset.y[i];
            }
        }
        return 0.0;
    }
    
    private static void printConfidence(String modelName, PerformanceMetrics metrics) {
        System.out.println("\n  📊 Model Confidence Metrics (Chỉ số độ tin cậy):");
        System.out.printf("    Train R²: %.2f%% | MAPE: %.2f%%\n", metrics.trainR2 * 100, metrics.trainMAPE);
        System.out.printf("    Val R²:   %.2f%% | MAPE: %.2f%%\n", metrics.valR2 * 100, metrics.valMAPE);
        System.out.printf("    Test R²:  %.2f%% | MAPE: %.2f%%\n", metrics.testR2 * 100, metrics.testMAPE);
        
        // Overall confidence score
        double avgR2 = (metrics.trainR2 + metrics.valR2 + metrics.testR2) / 3;
        double avgMAPE = (metrics.trainMAPE + metrics.valMAPE + metrics.testMAPE) / 3;
        
        System.out.printf("\n  ⭐ Overall Performance (Hiệu suất tổng thể):\n");
        System.out.printf("    Average R²: %.2f%%\n", avgR2 * 100);
        System.out.printf("    Average MAPE: %.2f%%\n", avgMAPE);
        
        // Confidence rating
        String rating;
        String emoji;
        if (avgR2 > 0.9 && avgMAPE < 10) {
            rating = "EXCELLENT (XUẤT SẮC)";
            emoji = "✅✅✅";
        } else if (avgR2 > 0.8 && avgMAPE < 15) {
            rating = "GOOD (TỐT)";
            emoji = "✅✅";
        } else if (avgR2 > 0.7 && avgMAPE < 20) {
            rating = "FAIR (KHÁ)";
            emoji = "✅";
        } else {
            rating = "NEEDS IMPROVEMENT (CẦN CẢI THIỆN)";
            emoji = "⚠️";
        }
        
        System.out.printf("    Confidence (Độ tin cậy): %s %s\n", rating, emoji);
        System.out.printf("    Expected error range (Phạm vi sai số dự kiến): ±%.2f%%\n", avgMAPE);
    }
    
    static class PerformanceMetrics {
        double trainR2, trainMAPE;
        double valR2, valMAPE;
        double testR2, testMAPE;
        
        PerformanceMetrics(double trainR2, double trainMAPE, 
                          double valR2, double valMAPE,
                          double testR2, double testMAPE) {
            this.trainR2 = trainR2;
            this.trainMAPE = trainMAPE;
            this.valR2 = valR2;
            this.valMAPE = valMAPE;
            this.testR2 = testR2;
            this.testMAPE = testMAPE;
        }
    }
}