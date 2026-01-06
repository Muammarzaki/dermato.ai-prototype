package com.github.dermatoai.data.db.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "prediction_records")
data class PredictionRecordEntity(
    // ----------------------------------------------------------------
    // Primary
    // ----------------------------------------------------------------

    @PrimaryKey(autoGenerate = true)
    val id: Long = 0L,

    // ----------------------------------------------------------------
    // Prediction Result (Business Data)
    // ----------------------------------------------------------------

    @ColumnInfo(name = "label")
    val label: String,

    @ColumnInfo(name = "confidence")
    val confidence: Float,

    @ColumnInfo(name = "all_results_json")
    val allResultsJson: String? = null,
    // JSON string (optional):
    // [{label:"Acne",confidence:0.88}, {...}]

    // ----------------------------------------------------------------
    // Image Info
    // ----------------------------------------------------------------

    @ColumnInfo(name = "image_uri")
    val imageUri: String,

    @ColumnInfo(name = "image_width")
    val imageWidth: Int? = null,

    @ColumnInfo(name = "image_height")
    val imageHeight: Int? = null,

    @ColumnInfo(name = "image_size_bytes")
    val imageSizeBytes: Long? = null,

    @ColumnInfo(name = "image_mime_type")
    val imageMimeType: String? = null,

    // ----------------------------------------------------------------
    // Network / Technical Info
    // ----------------------------------------------------------------

    @ColumnInfo(name = "protocol")
    val protocol: String, // REST / gRPC

    @ColumnInfo(name = "latency_ms")
    val latencyMs: Long? = null,

    @ColumnInfo(name = "http_status")
    val httpStatus: Int? = null,

    @ColumnInfo(name = "grpc_status")
    val grpcStatus: String? = null,

    // ----------------------------------------------------------------
    // Error & Debug
    // ----------------------------------------------------------------

    @ColumnInfo(name = "is_success")
    val isSuccess: Boolean = true,

    @ColumnInfo(name = "error_message")
    val errorMessage: String? = null,

    // ----------------------------------------------------------------
    // Audit / Lifecycle
    // ----------------------------------------------------------------

    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis()
)
