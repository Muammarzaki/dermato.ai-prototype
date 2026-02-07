package com.github.dermatoai.data.mapper

import com.github.dermatoai.data.db.entity.PredictionRecordEntity
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.entity.DiseaseResult
import com.github.dermatoai.domain.entity.ImageInfo
import com.github.dermatoai.domain.entity.PerformanceMetrics
import java.security.MessageDigest


fun DiagnosisSession.toEntity(): PredictionRecordEntity = PredictionRecordEntity(
    id = this.id.toLongOrNull() ?: 0L,
    label = this.disease.name,
    confidence = this.disease.confidence,
    latencyMs = this.metrics?.latencyMs,
    imageUri = this.image?.imageUri ?: "",
    imageWidth = this.image?.imageWidth,
    imageHeight = this.image?.imageHeight,
    imageSizeBytes = this.image?.imageSizeBytes,
    imageMimeType = this.image?.imageMimeType,
    protocol = this.metrics?.protocolUsed ?: "",
    httpStatus = null,
    grpcStatus = null,
    isSuccess = this.metrics?.status ?: false,
    errorMessage = null,
    createdAt = this.timestamp,
    imageSha256 = this.image?.imageSha256?.toHex() ?: "",
)


fun PredictionRecordEntity.toDomain(): DiagnosisSession {
    return DiagnosisSession(
        id = this.id.toString(),
        disease = DiseaseResult(name = this.label, confidence = this.confidence),
        metrics = PerformanceMetrics(
            latencyMs = this.latencyMs ?: 0L,
            protocolUsed = this.protocol,
            status = this.isSuccess
        ),
        timestamp = this.createdAt,
        image = ImageInfo(
            imageUri = this.imageUri,
            imageWidth = this.imageWidth,
            imageHeight = this.imageHeight,
            imageSizeBytes = this.imageSizeBytes,
            imageSha256 = this.imageSha256.toByteArray(),
            imageMimeType = this.imageMimeType
        )
    )
}

fun ByteArray.toHex(): String =
    joinToString("") { "%02x".format(it) }

fun ByteArray.sha256(): ByteArray {
    val digest = MessageDigest.getInstance("SHA-256")
    return digest.digest(this)
}