package com.github.dermatoai.ui.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.github.dermatoai.ui.screen.HomeScreen
import com.github.dermatoai.ui.theme.DermatoaiTheme
import com.github.dermatoai.ui.vm.AnalyzeVM
import com.github.dermatoai.ui.vm.DataVM
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val analyzeViewModel: AnalyzeVM by viewModels()

    private val dataViewModel: DataVM by viewModels()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DermatoaiTheme {
                HomeScreen(
                    analyzeVM = analyzeViewModel,
                    dataVM = dataViewModel
                )
            }
        }
    }
}