data_files = [
    {
        "name": "shd",
        "drop_columns": [],
        "encode_columns": ['presence'],
        "target": "presence",
        "sep": ",",
        "no_scale":['presence', 'sex'],
        "is_multiclass": False
    },
    {
        "name": "chd",
        "drop_columns": [],
        "encode_columns": ['condition'],
        "target": "condition",
        "sep": ",",
        "no_scale":['condition', 'sex'],
        "is_multiclass": False
    },
    {
        "name": "ilpd",
        "drop_columns": [],
        "encode_columns": ["a1", 'a10'],
        "target": "a10",
        "sep": ",",
        "no_scale":['a1', 'a10'],
        "is_multiclass":False
    },
    {
        "name": "vcd",
        "drop_columns": [],
        "encode_columns": ['A6'],
        "target": "A6",
        "sep": ",",
        "no_scale":['A6'],
        "is_multiclass":False
    },
    {
        "name": "lymph",
        "drop_columns": [],
        "encode_columns": ['a0'],
        "target": "a0",
        "sep": ",",
        "no_scale":['a0'],
        "is_multiclass":True
    },
    # {
    #     "name": "iono",
    #     "drop_columns": [],
    #     "encode_columns": ['a34'],
    #     "target": "a34",
    #     "sep": ",",
    #     "no_scale":['a0', 'a1', 'a34'],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "krkp",
    #     "drop_columns": [],
    #     "encode_columns": ['a36'],
    #     "target": "a36",
    #     "sep": ",",
    #     "no_scale":['a0', 'a36', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10''a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35'],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "iris",
    #     "drop_columns": ["Id"],
    #     "encode_columns": ["Species"],
    #     "target": "Species",
    #     "sep": ",",
    #     "no_scale": ["Species"],
    #     "is_multiclass": True
    # },
    # {
    #     "name": "parkinsons",
    #     "drop_columns": [],
    #     "encode_columns": ["name", 'status'],
    #     "target": "status",
    #     "sep": ",",
    #     "no_scale":["status"],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "cirrhosis",
    #     "drop_columns": ["ID"],
    #     "encode_columns": ["Drug", "Status", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"],
    #     "target": "Stage",
    #     "sep": ",",
    #     "no_scale":["Stage", "Drug", "Status", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"],
    #     "is_multiclass":True
    # },
    # {
    #     "name": "diabetes",
    #     "drop_columns": [],
    #     "encode_columns": ['Outcome'],
    #     "target": "Outcome",
    #     "sep": ",",
    #     "no_scale":["Outcome", "Age", "BMI", "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "heart_disease",
    #     "drop_columns": [],
    #     "encode_columns": ['condition'],
    #     "target": "condition",
    #     "sep": ",",
    #     "no_scale":["condition", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "sonar",
    #     "drop_columns": [],
    #     "encode_columns": ["Class"],
    #     "target": "Class",
    #     "sep": ",",
    #     "no_scale":["Class"],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "stroke",
    #     "drop_columns": ["id"],
    #     "encode_columns": ["gender", 'stroke', "ever_married", "work_type", "Residence_type", "smoking_status"],
    #     "target": "stroke",
    #     "sep": ",",
    #     "no_scale":["stroke", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    #     "is_multiclass":False
    # },
    # {
    #     "name": "abalone",
    #     "drop_columns": [],
    #     "encode_columns": ['Age Class'],
    #     "target": "Age Class",
    #     "sep": ",",
    #     "no_scale":["Age Class"],
    #     "is_multiclass":True
    # },
    # {
    #     "name": "wine",
    #     "drop_columns": [],
    #     "encode_columns": ['quality'],
    #     "target": "quality",
    #     "sep": ",",
    #     "no_scale": ["quality"],
    #     "is_multiclass": False
    # },
    # {
    #     "name": "breast",
    #     "drop_columns": ["id", "Unnamed: 32"],
    #     "encode_columns": ["diagnosis"],
    #     "target": "diagnosis",
    #     "sep": ",",
    #     "no_scale":["diagnosis"],
    #     "is_multiclass":False
    # }
]
