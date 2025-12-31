package com.github.dermatoai.domain.usecase

import com.github.dermatoai.data.repository.NetworkAnalyzeApiRepository
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.entity.DiagnosisSession
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject

class AnalyzeUseCaseImpl @Inject constructor(
    private val repository: NetworkAnalyzeApiRepository
) : AnalyzeUseCase {
    override fun invoke(
        imageUri: String,
        protocol: NetworkProtocol
    ): Flow<Resource<DiagnosisSession>> = flow {
        // 1. Emit Loading (UI munculkan loading)
        emit(Resource.Loading)

        try {
            // 2. Validasi Bisnis (Input check)
            if (imageUri.isBlank()) {
                throw IllegalArgumentException("Gambar tidak valid")
            }

            // 3. Panggil Repository (Simulasi request ke gRPC/AI)
//            val result = repository.predict(imageUri)
//
//            // 4. Logika Bisnis Tambahan (Opsional)
//            // Misal: Jika confidence terlalu rendah, anggap error atau warning
//            if (result.confidence < 0.2f) {
//                emit(Resource.Error("Hasil tidak meyakinkan, coba ambil foto ulang."))
//            } else {
//                emit(Resource.Success(result))
//            }

        } catch (e: Exception) {
            // 5. Tangkap Error (Jaringan putus, Server down)
            emit(Resource.Error(e.localizedMessage ?: "Terjadi kesalahan sistem"))
        }
    }
}