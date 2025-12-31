package com.github.dermatoai.domain.repository

import android.net.Uri
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.entity.DiagnosisSession

interface NetworkAnalyzeRepository {
    suspend fun predict(
        uri: Uri,
        protocol: NetworkProtocol
    ): DiagnosisSession
}