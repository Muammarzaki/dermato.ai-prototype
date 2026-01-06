package com.github.dermatoai.domain.entity

data class ImageInfo(
    val imageUri: String,
    val imageWidth: Int? = null,
    val imageHeight: Int? = null,
    val imageSizeBytes: Long? = null,
    val imageMimeType: String? = null,
)
