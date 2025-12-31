package com.github.dermatoai.domain.usecase

import com.github.dermatoai.data.repository.NetworkAnalyzeApiRepository
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.repository.LocalDBRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject

class AnalyzeUseCaseImpl @Inject constructor(
    private val netRepository: NetworkAnalyzeApiRepository,
    private val localRepository: LocalDBRepository
) : AnalyzeUseCase {
    override fun invoke(
        imageUri: String,
        protocol: NetworkProtocol
    ): Flow<Resource<DiagnosisSession>> = flow {
        emit(Resource.Loading)

        try {
            if (imageUri.isBlank()) {
                throw IllegalArgumentException("URI Gambar tidak valid")
            }

            val uriObj = android.net.Uri.parse(imageUri)
            val result = netRepository.predict(uriObj, protocol)

            localRepository.savePredictionResult(result)

            if (result.disease.confidence < 0.5f) {
                emit(Resource.Error("Hasil diagnosa kurang meyakinkan (${result.disease.confidence * 100}%)"))
            } else {
                emit(Resource.Success(result))
            }

        } catch (e: Exception) {
            emit(Resource.Error(e.localizedMessage ?: "Terjadi kesalahan sistem"))
        }
    }
}