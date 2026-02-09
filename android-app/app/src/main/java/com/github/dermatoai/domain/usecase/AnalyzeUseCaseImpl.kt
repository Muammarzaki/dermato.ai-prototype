package com.github.dermatoai.domain.usecase

import com.github.dermatoai.data.repository.NetworkAnalyzeApiRepository
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.repository.LocalDBRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import androidx.core.net.toUri

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
                throw IllegalArgumentException("Invalid image URI")
            }

            val uriObj = imageUri.toUri()
            val result = netRepository.predict(uriObj, protocol)

            localRepository.savePrediction(result)

            if (result.disease.confidence < 0.5f) {
                emit(Resource.Error("The diagnosis results were not conclusive enough (${result.disease.confidence * 100}%)"))
            } else {
                emit(Resource.Success(result))
            }

        } catch (e: Exception) {
            emit(Resource.Error(e.localizedMessage ?: "A system error occurred"))
        }
    }
}