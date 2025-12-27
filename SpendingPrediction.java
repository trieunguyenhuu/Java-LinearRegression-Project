import java.io.IOException;
import java.util.Scanner;

/**
 * Spending Prediction System - Next Month Forecast
 * Version 5.0: Predict next month for given Account_Key
 */
public class SpendingPrediction {
    
    private static LinearRegression model1; // Total_Monthly_Spend
    private static LinearRegression model2; // Frequency_Total
    private static LinearRegression model3; // Amount_Entertainment
    
    private static DataLoader.Dataset dataset1;
    private static DataLoader.Dataset dataset2;
    private static DataLoader.Dataset dataset3;
    
    private static SimpleScalerInfo scalerInfo; // ⭐ Thêm SimpleScalerInfo
    
    // Model files
    private static final String MODEL1_FILE = "model_total_spend_focused.dat";
    private static final String MODEL2_FILE = "model_frequency_focused.dat";
    private static final String MODEL3_FILE = "model_entertainment_focused.dat";
    
    // Data files
    private static final String DATA1_FILE = "customer_spending_cleaned_Y1_Total_Spend.csv";
    private static final String DATA2_FILE = "customer_spending_cleaned_Y2_Frequency.csv";
    private static final String DATA3_FILE = "customer_spending_cleaned_Y3_Entertainment.csv";
    
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(80));
            System.out.println("CUSTOMER SPENDING PREDICTION SYSTEM v5.0");
            System.out.println("Next Month Forecast - Dự báo tháng tới");
            System.out.println("=".repeat(80));
            System.out.println();
            
            // Load data
            System.out.println("[STEP 1] Loading datasets...");
            System.out.println("=".repeat(80));
            dataset1 = DataLoader.loadFromCSV(DATA1_FILE, "Total_Monthly_Spend");
            dataset2 = DataLoader.loadFromCSV(DATA2_FILE, "Frequency_Total");
            dataset3 = DataLoader.loadFromCSV(DATA3_FILE, "Amount_Entertainment");
            
            // ⭐ Initialize SimpleScalerInfo (không cần file JSON!)
            System.out.println("Initializing scaler info...");
            scalerInfo = new SimpleScalerInfo();
            scalerInfo.printInfo();
            
            System.out.println("=".repeat(80));
            System.out.println();
            
            // Check if models exist
            if (ModelSerializer.modelsExist(MODEL1_FILE, MODEL2_FILE, MODEL3_FILE)) {
                System.out.println("=".repeat(80));
                System.out.println("Phát hiện models đã được train trước đó!");
                System.out.println("=".repeat(80));
                System.out.println();
                
                try (Scanner scanner = new Scanner(System.in)) {
                    System.out.print("Bạn muốn:\n");
                    System.out.print("  [1] Load models có sẵn (nhanh)\n");
                    System.out.print("  [2] Train lại từ đầu\n");
                    System.out.print("Chọn (1/2): ");
                    
                    String choice = scanner.nextLine().trim();
                    System.out.println();
                    
                    if (choice.equals("1")) {
                        loadModels();
                    } else {
                        trainAndSaveModels();
                    }
                }
            } else {
                System.out.println("Chưa có models được train. Đang bắt đầu training...\n");
                trainAndSaveModels();
            }
            
            // Interactive prediction
            interactivePrediction();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Train and save models
     */
    private static void trainAndSaveModels() throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("TRAINING MODELS");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // Split datasets
        System.out.println("[STEP 2] Splitting datasets (80% train, 20% test)...");
        DataLoader.Dataset[] splits1 = DataLoader.splitTrainTest(dataset1, 0.8);
        DataLoader.Dataset[] splits2 = DataLoader.splitTrainTest(dataset2, 0.8);
        DataLoader.Dataset[] splits3 = DataLoader.splitTrainTest(dataset3, 0.8);
        
        DataLoader.Dataset train1 = splits1[0], test1 = splits1[1];
        DataLoader.Dataset train2 = splits2[0], test2 = splits2[1];
        DataLoader.Dataset train3 = splits3[0], test3 = splits3[1];
        
        System.out.println("Dataset 1 (Total Spend): Train=" + train1.X.length + ", Test=" + test1.X.length);
        System.out.println("Dataset 2 (Frequency): Train=" + train2.X.length + ", Test=" + test2.X.length);
        System.out.println("Dataset 3 (Entertainment): Train=" + train3.X.length + ", Test=" + test3.X.length);
        System.out.println();
        
        // Train models with Ridge Regularization
        System.out.println("[STEP 3] Training models...");
        System.out.println("=".repeat(80));
        
        System.out.println("\n>>> Model 1: Total Monthly Spend <<<");
        model1 = new LinearRegression("Total_Monthly_Spend", 0.1);
        model1.train(train1.X, train1.y);
        
        System.out.println("\n>>> Model 2: Transaction Frequency <<<");
        model2 = new LinearRegression("Frequency_Total", 0.1);
        model2.train(train2.X, train2.y);
        
        System.out.println("\n>>> Model 3: Entertainment Spending <<<");
        model3 = new LinearRegression("Amount_Entertainment", 0.1);
        model3.train(train3.X, train3.y);
        
        System.out.println();
        System.out.println("=".repeat(80));
        
        // Evaluate
        System.out.println("\n[STEP 4] Evaluating models...");
        System.out.println("=".repeat(80));
        
        evaluateModel(model1, train1.X, train1.y, test1.X, test1.y);
        System.out.println();
        
        evaluateModel(model2, train2.X, train2.y, test2.X, test2.y);
        System.out.println();
        
        evaluateModel(model3, train3.X, train3.y, test3.X, test3.y);
        System.out.println();
        
        System.out.println("=".repeat(80));
        
        // Save models
        System.out.println("\n[STEP 5] Saving models...");
        System.out.println("-".repeat(80));
        ModelSerializer.saveModel(model1, MODEL1_FILE);
        ModelSerializer.saveModel(model2, MODEL2_FILE);
        ModelSerializer.saveModel(model3, MODEL3_FILE);
        System.out.println("-".repeat(80));
        System.out.println("✓ Models saved!");
        System.out.println();
        
        System.out.println("=".repeat(80));
        System.out.println("TRAINING COMPLETED!");
        System.out.println("=".repeat(80));
        System.out.println();
    }
    
    /**
     * Load pre-trained models
     */
    private static void loadModels() throws IOException, ClassNotFoundException {
        System.out.println("=".repeat(80));
        System.out.println("LOADING PRE-TRAINED MODELS");
        System.out.println("=".repeat(80));
        System.out.println();
        
        model1 = ModelSerializer.loadModel("Total_Monthly_Spend", MODEL1_FILE);
        model2 = ModelSerializer.loadModel("Frequency_Total", MODEL2_FILE);
        model3 = ModelSerializer.loadModel("Amount_Entertainment", MODEL3_FILE);
        
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("✓ MODELS LOADED!");
        System.out.println("=".repeat(80));
        System.out.println();
    }
    
    /**
     * Interactive prediction - Predict next month
     */
    private static void interactivePrediction() {
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.println();
            System.out.println("=".repeat(80));
            System.out.println("DỰ BÁO CHI TIÊU THÁNG TỚI");
            System.out.println("=".repeat(80));
            
            System.out.print("Nhập Account_Key (hoặc 'exit' để thoát): ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) {
                System.out.println("\nĐã thoát chương trình. Tạm biệt!");
                break;
            }
            
            try {
                int accountKey = Integer.parseInt(input);
                predictNextMonth(accountKey);
                
            } catch (NumberFormatException e) {
                System.out.println("❌ Lỗi: Vui lòng nhập số nguyên hợp lệ!");
            }
        }
        
        scanner.close();
    }
    
    /**
     * Predict next month for given account
     */
    private static void predictNextMonth(int accountKey) {
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("KẾT QUẢ DỰ BÁO THÁNG TỚI");
        System.out.println("=".repeat(80));
        System.out.println("Account_Key: " + accountKey);
        System.out.println("=".repeat(80));
        
        // Get latest data for this account from each dataset
        Integer latestIndex1 = dataset1.getLatestIndexByAccount(accountKey);
        Integer latestIndex2 = dataset2.getLatestIndexByAccount(accountKey);
        Integer latestIndex3 = dataset3.getLatestIndexByAccount(accountKey);
        
        if (latestIndex1 == null && latestIndex2 == null && latestIndex3 == null) {
            System.out.println("\n❌ Không tìm thấy dữ liệu cho Account_Key: " + accountKey);
            System.out.println("=".repeat(80));
            return;
        }
        
        // Get latest time info
        int latestYear = 0, latestMonth = 0;
        if (latestIndex1 != null) {
            latestYear = dataset1.years[latestIndex1];
            latestMonth = dataset1.months[latestIndex1];
        } else if (latestIndex2 != null) {
            latestYear = dataset2.years[latestIndex2];
            latestMonth = dataset2.months[latestIndex2];
        } else if (latestIndex3 != null) {
            latestYear = dataset3.years[latestIndex3];
            latestMonth = dataset3.months[latestIndex3];
        }
        
        // Calculate next month
        int nextYear = latestYear;
        int nextMonth = latestMonth + 1;
        if (nextMonth > 12) {
            nextMonth = 1;
            nextYear++;
        }
        
        System.out.println("Dữ liệu gần nhất: " + latestYear + "/" + latestMonth);
        System.out.println("Dự báo cho: " + nextYear + "/" + nextMonth);
        System.out.println("=".repeat(80));
        
        boolean foundAny = false;
        
        // Y1: Total Monthly Spend
        if (latestIndex1 != null) {
            foundAny = true;
            double[] features1 = dataset1.X[latestIndex1];
            double predNormalized = model1.predictSingle(features1);
            
            // ⭐ Denormalize prediction
            double predActual = scalerInfo.denormalize("Total_Monthly_Spend", predNormalized);
            
            System.out.println("\n[1] TOTAL MONTHLY SPEND (Tổng chi tiêu tháng)");
            System.out.println("-".repeat(80));
            System.out.printf("  Dự báo cho %d/%d: %s\n", 
                nextYear, nextMonth, 
                scalerInfo.getScaler("Total_Monthly_Spend").formatValue(predActual));
            
            // Show context from latest month
            double currentNormalized = dataset1.y[latestIndex1];
            double currentActual = scalerInfo.denormalize("Total_Monthly_Spend", currentNormalized);
            
            System.out.printf("  Tháng hiện tại %d/%d: %s\n", 
                latestYear, latestMonth,
                scalerInfo.getScaler("Total_Monthly_Spend").formatValue(currentActual));
            
            double change1 = ((predActual - currentActual) / currentActual) * 100;
            if (change1 > 0) {
                System.out.printf("  Xu hướng: TĂNG %.2f%% (+%s)\n", 
                    change1,
                    scalerInfo.getScaler("Total_Monthly_Spend").formatValue(predActual - currentActual));
            } else {
                System.out.printf("  Xu hướng: GIẢM %.2f%% (%s)\n", 
                    Math.abs(change1),
                    scalerInfo.getScaler("Total_Monthly_Spend").formatValue(predActual - currentActual));
            }
        }
        
        // Y2: Frequency
        if (latestIndex2 != null) {
            foundAny = true;
            double[] features2 = dataset2.X[latestIndex2];
            double predNormalized = model2.predictSingle(features2);
            
            // ⭐ Denormalize prediction
            double predActual = scalerInfo.denormalize("Frequency_Total", predNormalized);
            
            System.out.println("\n[2] TRANSACTION FREQUENCY (Tần suất giao dịch)");
            System.out.println("-".repeat(80));
            System.out.printf("  Dự báo cho %d/%d: %s\n", 
                nextYear, nextMonth,
                scalerInfo.getScaler("Frequency_Total").formatValue(predActual));
            
            double currentNormalized = dataset2.y[latestIndex2];
            double currentActual = scalerInfo.denormalize("Frequency_Total", currentNormalized);
            
            System.out.printf("  Tháng hiện tại %d/%d: %s\n", 
                latestYear, latestMonth,
                scalerInfo.getScaler("Frequency_Total").formatValue(currentActual));
            
            double change2 = ((predActual - currentActual) / currentActual) * 100;
            if (change2 > 0) {
                System.out.printf("  Xu hướng: TĂNG %.2f%% (+%s)\n", 
                    change2,
                    scalerInfo.getScaler("Frequency_Total").formatValue(predActual - currentActual));
            } else {
                System.out.printf("  Xu hướng: GIẢM %.2f%% (%s)\n", 
                    Math.abs(change2),
                    scalerInfo.getScaler("Frequency_Total").formatValue(predActual - currentActual));
            }
        }
        
        // Y3: Entertainment
        if (latestIndex3 != null) {
            foundAny = true;
            double[] features3 = dataset3.X[latestIndex3];
            double predNormalized = model3.predictSingle(features3);
            
            // ⭐ Denormalize prediction
            double predActual = scalerInfo.denormalize("Amount_Entertainment", predNormalized);
            
            System.out.println("\n[3] ENTERTAINMENT SPENDING (Chi tiêu giải trí)");
            System.out.println("-".repeat(80));
            System.out.printf("  Dự báo cho %d/%d: %s\n", 
                nextYear, nextMonth,
                scalerInfo.getScaler("Amount_Entertainment").formatValue(predActual));
            
            double currentNormalized = dataset3.y[latestIndex3];
            double currentActual = scalerInfo.denormalize("Amount_Entertainment", currentNormalized);
            
            System.out.printf("  Tháng hiện tại %d/%d: %s\n", 
                latestYear, latestMonth,
                scalerInfo.getScaler("Amount_Entertainment").formatValue(currentActual));
            
            double change3 = ((predActual - currentActual) / currentActual) * 100;
            if (change3 > 0) {
                System.out.printf("  Xu hướng: TĂNG %.2f%% (+%s)\n", 
                    change3,
                    scalerInfo.getScaler("Amount_Entertainment").formatValue(predActual - currentActual));
            } else {
                System.out.printf("  Xu hướng: GIẢM %.2f%% (%s)\n", 
                    Math.abs(change3),
                    scalerInfo.getScaler("Amount_Entertainment").formatValue(predActual - currentActual));
            }
        }
        
        if (!foundAny) {
            System.out.println("\n❌ Không có dữ liệu để dự báo");
        }
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Evaluate model
     */
    private static void evaluateModel(LinearRegression model, 
                                     double[][] X_train, double[] y_train,
                                     double[][] X_test, double[] y_test) {
        System.out.println(">>> " + model.getModelName() + " <<<");
        
        double testMSE = model.calculateMSE(X_test, y_test);
        double testRMSE = Math.sqrt(testMSE);
        double testR2 = model.calculateR2(X_test, y_test);
        
        System.out.printf("Test Performance:\n");
        System.out.printf("  MSE:  %.6f\n", testMSE);
        System.out.printf("  RMSE: %.6f\n", testRMSE);
        System.out.printf("  R²:   %.4f (%.2f%%)\n", testR2, testR2 * 100);
        
        // MAPE
        double[] predictions = model.predict(X_test);
        double mape = 0;
        int validCount = 0;
        for (int i = 0; i < y_test.length; i++) {
            if (Math.abs(y_test[i]) > 1e-6) {
                mape += Math.abs((y_test[i] - predictions[i]) / y_test[i]);
                validCount++;
            }
        }
        if (validCount > 0) {
            mape = (mape / validCount) * 100;
            System.out.printf("  MAPE: %.2f%%\n", mape);
        }
    }
}