package com.github.dermatoai.ui.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.github.dermatoai.ui.screen.HomeScreen
import com.github.dermatoai.ui.theme.DermatoaiTheme
import com.github.dermatoai.ui.vm.AnalyzeVM
import com.github.dermatoai.ui.vm.DataVM
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject
    lateinit var analyzeViewModel: AnalyzeVM

    @Inject
    lateinit var dataViewModel: DataVM


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