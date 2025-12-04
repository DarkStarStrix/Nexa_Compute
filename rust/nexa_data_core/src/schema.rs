use std::collections::HashMap;
use arrow::datatypes::{DataType, Schema};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FieldDef {
    pub name: String,
    pub dtype: String, // "utf8", "int64", "float64", etc.
    pub nullable: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchemaDef {
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("Field missing: {0}")]
    FieldMissing(String),
    #[error("Type mismatch for field {0}: expected {1}, got {2}")]
    TypeMismatch(String, String, String),
    #[error("Nullable mismatch for field {0}: expected {1}, got {2}")]
    NullableMismatch(String, bool, bool),
}

impl From<SchemaError> for PyErr {
    fn from(err: SchemaError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

pub fn validate_arrow_schema(arrow_schema: &Schema, schema_def: &SchemaDef) -> Result<(), SchemaError> {
    let arrow_fields: HashMap<String, &arrow::datatypes::Field> = arrow_schema.fields().iter()
        .map(|f| (f.name().clone(), f.as_ref()))
        .collect();

    for field_def in &schema_def.fields {
        let arrow_field = arrow_fields.get(&field_def.name)
            .ok_or_else(|| SchemaError::FieldMissing(field_def.name.clone()))?;

        let expected_type = parse_dtype(&field_def.dtype);
        if arrow_field.data_type() != &expected_type {
            return Err(SchemaError::TypeMismatch(
                field_def.name.clone(),
                format!("{:?}", expected_type),
                format!("{:?}", arrow_field.data_type())
            ));
        }

        if arrow_field.is_nullable() != field_def.nullable {
             return Err(SchemaError::NullableMismatch(
                field_def.name.clone(),
                field_def.nullable,
                arrow_field.is_nullable()
            ));
        }
    }

    Ok(())
}

fn parse_dtype(dtype: &str) -> DataType {
    match dtype {
        "utf8" | "string" => DataType::Utf8,
        "int64" | "long" => DataType::Int64,
        "int32" | "int" => DataType::Int32,
        "float64" | "double" => DataType::Float64,
        "float32" | "float" => DataType::Float32,
        "bool" | "boolean" => DataType::Boolean,
        _ => DataType::Utf8, // Default or error? defaulting for now
    }
}

#[pyfunction]
pub fn validate_schema(arrow_schema_json: String, schema_def_json: String) -> PyResult<()> {
    // This function is a bit tricky without Arrow-PyO3 interop fully set up in this snippet.
    // Ideally, we'd receive a PyArrow Schema object. 
    // For simplicity in this phase, we assume the caller passes JSON representations 
    // or we implement strict binding later.
    // Let's placeholder this as a validation logic unit.
    Ok(())
}
