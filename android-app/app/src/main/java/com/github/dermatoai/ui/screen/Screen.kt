package com.github.dermatoai.ui.screen

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.ui.graphics.vector.ImageVector

sealed class Screen(val route: String, val title: String, val valicon: ImageVector) {
    object Analyze : Screen("analyze", "Scan", Icons.Default.ImageSearch)
    object History : Screen("history", "History", Icons.Default.History)
    object Stats : Screen("stats", "Metrics", Icons.Default.Analytics)
}