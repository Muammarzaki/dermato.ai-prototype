package com.github.dermatoai.domain.entity

data class ImageInfo(
    val imageUri: String,
    val imageSha256: ByteArray,
    val imageWidth: Int? = null,
    val imageHeight: Int? = null,
    val imageSizeBytes: Long? = null,
    val imageMimeType: String? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ImageInfo

        if (imageWidth != other.imageWidth) return false
        if (imageHeight != other.imageHeight) return false
        if (imageSizeBytes != other.imageSizeBytes) return false
        if (imageUri != other.imageUri) return false
        if (!imageSha256.contentEquals(other.imageSha256)) return false
        if (imageMimeType != other.imageMimeType) return false

        return true
    }

    override fun hashCode(): Int {
        var result = imageWidth ?: 0
        result = 31 * result + (imageHeight ?: 0)
        result = 31 * result + (imageSizeBytes?.hashCode() ?: 0)
        result = 31 * result + imageUri.hashCode()
        result = 31 * result + imageSha256.contentHashCode()
        result = 31 * result + (imageMimeType?.hashCode() ?: 0)
        return result
    }
}
