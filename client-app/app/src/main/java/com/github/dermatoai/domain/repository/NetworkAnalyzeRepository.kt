package com.github.dermatoai.domain.repository

import android.net.Uri
import com.github.dermatoai.domain.enum.NetworkProtocol
import com.github.dermatoai.ui.screen.PredictionHistory

interface NetworkAnalyzeRepository {
    suspend fun predict(
        uri: Uri,
        protocol: NetworkProtocol
    ): PredictionHistory
}