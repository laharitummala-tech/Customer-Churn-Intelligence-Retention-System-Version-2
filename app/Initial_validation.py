import pandas as pd
import numpy as np
class InitialValidation:
    def __init__(self):
        self.expected_columns = [
            "Age", "Gender", "Country", "City", "Membership_Years",
            "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
            "Cart_Abandonment_Rate", "Wishlist_Items", "Total_Purchases",
            "Average_Order_Value", "Days_Since_Last_Purchase",
            "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
            "Customer_Service_Calls", "Product_Reviews_Written",
            "Social_Media_Engagement_Score", "Mobile_App_Usage",
            "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance",
            "Signup_Quarter"
        ]
        
        self.rules = {
            "Age < 18 or Age > 90": lambda df: ((df["Age"] < 18) | (df["Age"] > 90)),
            "Total_Purchases < 0": lambda df: (df["Total_Purchases"] < 0),
            "Cart_Abandonment_Rate > 100": lambda df: (df["Cart_Abandonment_Rate"] > 100),
            "Discount_Usage_Rate > 100": lambda df: (df["Discount_Usage_Rate"] > 100),
            "Returns_Rate > 100": lambda df: (df["Returns_Rate"] > 100),
            "Email_Open_Rate > 100": lambda df: (df["Email_Open_Rate"] > 100),
            "Customer_Service_Calls < 0": lambda df: (df["Customer_Service_Calls"] < 0),
            "Wishlist_Items < 0": lambda df: (df["Wishlist_Items"] < 0),
            "Product_Reviews_Written < 0": lambda df: (df["Product_Reviews_Written"] < 0),
            "Mobile_App_Usage < 0": lambda df: (df["Mobile_App_Usage"] < 0),
            "Lifetime_Value < 0": lambda df: (df["Lifetime_Value"] < 0),
            "Credit_Balance < 0": lambda df: (df["Credit_Balance"] < 0)
        }
        
        
    def remove_target_col(self,df):
        df = df.copy()
        if "Churned" in df.columns:
            df = df.drop("Churned",axis=1)
        return df
    
    def col_validation(self,df):
        missing_cols = [c for c in self.expected_columns if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in self.expected_columns]
        
        return missing_cols, extra_cols
    
    def missing_report(self,df):
        report = pd.DataFrame({
            "Column": df.columns,
            "Missing_count": df.isna().sum().values,
            "Missing_percentage": (df.isnull().mean()* 100).values
        })
        
        return report[report["Missing_count"] > 0].reset_index(drop=True)
    
    def invalid_data_report(self,df):
        report = []
        for rules_label, rules_action in self.rules.items():
            try:
                count = int(rules_action(df).sum())
                if count > 0:
                    report.append({"INVALID RULE":rules_label,"INVALID COUNT":count})
                    
            except KeyError:
                continue
        return pd.DataFrame(report)
                    
        
        
    def convert_invalid_to_nan(self,df):
        # here Im converting ll invalid values to nan and handled them in missing values
        df = df.copy()
        for rules_label, rules_action in self.rules.items():
            try:
                col_name = rules_label.split()[0]
                mask = rules_action(df)
                df.loc[mask, col_name] = np.nan
            except KeyError:
                continue
            
        return df
        
        
    def validate(self,df):
        df = self.remove_target_col(df)
        missing_cols,extra_cols = self.col_validation(df)
        
        result = {
            "clean_df": df,
            "missing_cols": missing_cols,
            "extra_cols": extra_cols,
            "missing_report": pd.DataFrame(),
            "invalid_data_report":  pd.DataFrame(),
            "is_valid" : True,
            "message" : "Validation comleted successfully"
        }
        
        if len(missing_cols) > 0:
            result["is_valid"] = False
            result["message"] =f"{len(missing_cols)} required column(s) missing: {', '.join(missing_cols)}"
            return result
        
        invalid_df = self.invalid_data_report(df)
        result["invalid_data_report"] = invalid_df
        
        
        # combinely show invalid values and missing values in the message
        df = self.convert_invalid_to_nan(df)
        result["clean_df"] = df[self.expected_columns]
        result["missing_report"] = self.missing_report(df)
        
        if not invalid_df.empty:
            result["message"] = (
                f"{len(invalid_df)} rule(s) violated — "
                f"invalid values converted to NaN and will be imputed automatically."
            )
            
        return result
        